"""Standalone MolmoAct2 action expert — extracted verbatim from the LeRobot HF
modeling (lerobot/policies/molmoact2/molmoact2_hf_model/modeling_molmoact2.py,
Apache-2.0, AI2 + HF). Only the action-expert classes are vendored here (pure
torch, no transformers dependency) so the coreai_torch (.venv) converter can build
and export the deployable `ActionExpert.forward` graph without loading the 5.44B VLM.

The deployable graph is one flow-matching denoise step: given noisy actions +
timestep + the VLM's per-layer K/V (encoder_kv_states), produce the velocity. The
host owns the VLM prefill (collect_layer_kv_states), the Euler loop
(trajectory += dt*velocity, num_flow_timesteps steps) and un-normalization.

Verified (reading the upstream): RoPE is real cos/sin (no complex dtype),
causal_attn=False, and ActionExpert.forward(modulation=None) is mathematically
identical to the loop's per-step forward_with_context (the block/final compute
`self.modulation(conditioning).chunk(...)` internally when modulation is None,
which is exactly what prepare_modulation_cache precomputes)."""

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _round_up_multiple(value: int, multiple_of: int) -> int:
    if multiple_of <= 0:
        return value
    return int(math.ceil(value / multiple_of) * multiple_of)


@dataclass
class ActionExpertContext:
    kv_contexts: Sequence[tuple[torch.Tensor, torch.Tensor]]
    cross_mask: torch.Tensor | None
    self_mask: torch.Tensor | None
    valid_action: torch.Tensor | None
    rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None


class ActionExpertRMSNorm(nn.Module):
    def __init__(self, size: int, *, eps: float = 1e-6, elementwise_affine: bool = False, device=None) -> None:
        super().__init__()
        self.size = size
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(size, device=device))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            dtype = x.dtype
            x_float = x.to(torch.float32)
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            out = x_float * torch.rsqrt(variance + self.eps)
            out = out.to(dtype)
        if self.weight is not None:
            out = out * self.weight
        return out


class ActionExpertRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")
        self.head_dim = head_dim
        self.base = base

    def build_cache(self, *, seq_len: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / max(half_dim, 1)))
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos().to(dtype=dtype).view(1, 1, seq_len, half_dim)
        sin = freqs.sin().to(dtype=dtype).view(1, 1, seq_len, half_dim)
        return cos, sin

    def forward(self, q, k, *, rope_cache=None):
        if rope_cache is None:
            rope_cache = self.build_cache(seq_len=q.shape[-2], device=q.device, dtype=q.dtype)
        cos, sin = rope_cache
        half_dim = self.head_dim // 2

        def _apply(x):
            x1, x2 = x[..., :half_dim], x[..., half_dim:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return _apply(q), _apply(k)


class ActionExpertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, *, attn_dropout=0.0, proj_dropout=0.0,
                 qk_norm=True, qk_norm_eps=1e-6, use_rope=True) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout
        self.q_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.rope = ActionExpertRotaryEmbedding(self.head_dim) if use_rope else None
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    def _apply_qk_norm(self, q, k):
        if self.q_norm is None or self.k_norm is None:
            return q, k
        return self.q_norm(q), self.k_norm(k)

    def _attention(self, q, k, v, *, attn_mask=None, is_causal=False):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)
        return out.transpose(1, 2).contiguous()

    def forward(self, x, *, attn_mask=None, is_causal=False, rope_cache=None):
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].contiguous()
        q, k = self._apply_qk_norm(q, k)
        if self.rope is not None:
            q, k = self.rope(q, k, rope_cache=rope_cache)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        out = self._attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.reshape(bsz, seq_len, self.hidden_size)
        return self.out_drop(self.out_proj(out))


class ActionExpertCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, *, attn_dropout=0.0, proj_dropout=0.0,
                 qk_norm=True, qk_norm_eps=1e-6) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout
        self.q_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    def _as_heads(self, x):
        if x.dim() == 4:
            if x.shape[2] == self.num_heads:
                return x
            if x.shape[1] == self.num_heads:
                return x.transpose(1, 2).contiguous()
            raise ValueError(f"Unexpected cross-attention KV shape {tuple(x.shape)}")
        if x.dim() != 3:
            raise ValueError(f"Expected 3D/4D cross-attention KV, got {tuple(x.shape)}")
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim)

    def _attention(self, q, k, v, *, attn_mask=None):
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        return out.transpose(1, 2).contiguous()

    def forward(self, x, *, kv_k, kv_v, attn_mask=None):
        bsz, tgt_len, _ = x.shape
        q = self.q_proj(x).view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = self._as_heads(kv_k)
        v = self._as_heads(kv_v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        out = self._attention(q, k, v, attn_mask=attn_mask)
        out = out.reshape(bsz, tgt_len, self.hidden_size)
        return self.out_drop(self.out_proj(out))


class ActionExpertMLP(nn.Module):
    def __init__(self, hidden_size, *, mlp_ratio, multiple_of, dropout=0.0) -> None:
        super().__init__()
        inner_dim = _round_up_multiple(int(hidden_size * mlp_ratio), multiple_of)
        self.up_proj = nn.Linear(hidden_size, inner_dim)
        self.gate_proj = nn.Linear(hidden_size, inner_dim)
        self.down_proj = nn.Linear(inner_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return self.dropout(x)


class ActionExpertModulation(nn.Module):
    def __init__(self, hidden_size, num_chunks) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(hidden_size, num_chunks * hidden_size)

    def forward(self, conditioning):
        return self.linear(self.act(conditioning))


class ActionExpertBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, *, mlp_ratio, ffn_multiple_of,
                 attn_dropout=0.0, dropout=0.0, qk_norm=True, qk_norm_eps=1e-6, rope=True) -> None:
        super().__init__()
        self.self_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.cross_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.ff_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.self_attn = ActionExpertSelfAttention(hidden_size, num_heads, attn_dropout=attn_dropout,
                                                   proj_dropout=dropout, qk_norm=qk_norm, qk_norm_eps=qk_norm_eps, use_rope=rope)
        self.cross_attn = ActionExpertCrossAttention(hidden_size, num_heads, attn_dropout=attn_dropout,
                                                    proj_dropout=dropout, qk_norm=qk_norm, qk_norm_eps=qk_norm_eps)
        self.mlp = ActionExpertMLP(hidden_size, mlp_ratio=mlp_ratio, multiple_of=ffn_multiple_of, dropout=dropout)
        self.modulation = ActionExpertModulation(hidden_size, 9)

    def forward(self, x, conditioning, *, cross_kv, self_attn_mask=None, attn_mask=None,
                is_causal=False, modulation=None, rope_cache=None):
        if modulation is None:
            modulation = self.modulation(conditioning).chunk(9, dim=1)
        (shift_msa, scale_msa, gate_msa, shift_mca, scale_mca, gate_mca,
         shift_mlp, scale_mlp, gate_mlp) = modulation
        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            _modulate(self.self_norm(x), shift_msa, scale_msa),
            attn_mask=self_attn_mask, is_causal=is_causal, rope_cache=rope_cache)
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(
            _modulate(self.cross_norm(x), shift_mca, scale_mca),
            kv_k=cross_kv[0], kv_v=cross_kv[1], attn_mask=attn_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.ff_norm(x), shift_mlp, scale_mlp))
        return x


class ActionExpertFinalLayer(nn.Module):
    def __init__(self, hidden_size, output_dim) -> None:
        super().__init__()
        self.norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.modulation = ActionExpertModulation(hidden_size, 2)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, x, conditioning, *, modulation=None):
        if modulation is None:
            modulation = self.modulation(conditioning).chunk(2, dim=1)
        shift, scale = modulation
        return self.linear(_modulate(self.norm(x), shift, scale))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        if timesteps.dim() > 1:
            timesteps = timesteps.view(timesteps.shape[0], -1)[:, 0]
        half_dim = self.dim // 2
        freq = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype)
                         * (-math.log(10000.0) / max(half_dim - 1, 1)))
        args = timesteps[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


@dataclass
class AEConfig:
    max_action_horizon: int = 32
    max_action_dim: int = 32
    hidden_size: int = 768
    num_layers: int = 36
    num_heads: int = 8
    timestep_embed_dim: int = 256
    mlp_ratio: float = 4.0
    ffn_multiple_of: int = 256
    attn_dropout: float = 0.0
    dropout: float = 0.0
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    rope: bool = True
    causal_attn: bool = False
    context_layer_norm: bool = True


class ActionExpert(nn.Module):
    """Per-layer conditioning: one action block per LLM layer, cross-attending to
    that layer's projected K/V. Vendored from the upstream (unused training-only
    paths dropped)."""

    def __init__(self, config: AEConfig, *, llm_kv_dim: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.llm_kv_dim = llm_kv_dim
        self.action_head_dim = config.hidden_size // config.num_heads
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(config.timestep_embed_dim),
            nn.Linear(config.timestep_embed_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size))
        self.action_embed = nn.Linear(config.max_action_dim, config.hidden_size)
        self.context_k_proj = nn.Linear(llm_kv_dim, config.hidden_size, bias=False)
        self.context_v_proj = nn.Linear(llm_kv_dim, config.hidden_size, bias=False)
        self.context_norm = ActionExpertRMSNorm(config.hidden_size, eps=1e-6) if config.context_layer_norm else nn.Identity()
        self.blocks = nn.ModuleList([
            ActionExpertBlock(config.hidden_size, config.num_heads, mlp_ratio=config.mlp_ratio,
                              ffn_multiple_of=config.ffn_multiple_of, attn_dropout=config.attn_dropout,
                              dropout=config.dropout, qk_norm=config.qk_norm, qk_norm_eps=config.qk_norm_eps,
                              rope=config.rope)
            for _ in range(config.num_layers)])
        self.final_layer = ActionExpertFinalLayer(config.hidden_size, config.max_action_dim)

    def _reshape_hidden_to_heads(self, x):
        return x.view(x.shape[0], x.shape[1], self.config.num_heads, self.action_head_dim)

    def _time_conditioning(self, timesteps):
        conditioning = self.time_embed[0](timesteps)
        first_linear = self.time_embed[1]
        if isinstance(first_linear, nn.Linear):
            conditioning = conditioning.to(dtype=first_linear.weight.dtype)
        for module in list(self.time_embed.children())[1:]:
            conditioning = module(conditioning)
        return conditioning

    def _project_kv_tensor(self, x, proj):
        flat = self.context_norm(proj(x))
        return self._reshape_hidden_to_heads(flat)

    def _prepare_kv_context(self, encoder_kv_states):
        kv_contexts = []
        for block, (k_in, v_in) in zip(self.blocks, encoder_kv_states):
            k_ctx = self._project_kv_tensor(k_in, self.context_k_proj)
            v_ctx = self._project_kv_tensor(v_in, self.context_v_proj)
            k_norm = block.cross_attn.k_norm
            if k_norm is not None:
                k_ctx = k_norm(k_ctx.transpose(1, 2)).transpose(1, 2)
            kv_contexts.append((k_ctx, v_ctx))
        return kv_contexts

    def forward(self, actions, timesteps, *, encoder_kv_states):
        """One flow-matching denoise step -> velocity [B, horizon, max_action_dim].
        encoder_kv_states: list of (k, v) per layer, each [B, ctx_seq, llm_kv_dim].
        Masks are None (causal_attn=False, all context valid); the host controls
        which VLM tokens are exposed via the K/V it prefills."""
        seq_len = actions.shape[1]
        rope_cache = None
        if len(self.blocks) > 0 and self.blocks[0].self_attn.rope is not None:
            rope_cache = self.blocks[0].self_attn.rope.build_cache(
                seq_len=seq_len, device=actions.device, dtype=actions.dtype)
        kv_contexts = self._prepare_kv_context(encoder_kv_states)
        conditioning = self._time_conditioning(timesteps)
        x = self.action_embed(actions)
        for block, kv_context in zip(self.blocks, kv_contexts):
            x = block(x, conditioning, cross_kv=kv_context, self_attn_mask=None,
                      attn_mask=None, is_causal=self.config.causal_attn, rope_cache=rope_cache)
        return self.final_layer(x, conditioning)
