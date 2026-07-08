"""Standalone LingBot-VLA 2.0 action expert — reconstructed from robbyant/lingbot-vla-v2
(Apache-2.0). Only the qwen_expert (action) path is vendored; the Qwen3-VL-4B backbone
is host-side. The deployable graph is `predict_velocity`: one flow-matching denoise
step over the 36-layer MoE expert stack, conditioned on the VLM's cached per-layer
prefix K/V (fed as graph inputs — the EO-1/MolmoAct2 split).

Faithfulness: the norm (AdaRMSNorm), eager attention, decoder-layer wiring and
embed_suffix are copied verbatim from the upstream; the MoE combine is rewritten as a
dense einsum over the checkpoint's fused 3D expert weights (validated exact vs the
sparse top-4 reference, cosine 1.0). Attention is eager (matches the deploy policy) —
manual softmax matmul, no SDPA, so the macOS-27 mask-free segfault cannot occur."""
import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LVConfig:
    num_layers: int = 36
    hidden: int = 768
    head_dim: int = 128
    n_q_heads: int = 32
    n_kv_heads: int = 8
    num_experts: int = 32
    top_k: int = 4
    expert_inter: int = 512
    shared_inter: int = 704
    routed_scaling: float = 4.0
    norm_topk_prob: bool = True
    router_activation: str = "sigmoid"
    action_dim: int = 55
    state_dim: int = 55
    proj_width: int = 768
    cond_dim: int = 768          # time-embedding dim driving AdaRMSNorm
    n_action_steps: int = 32     # denoise horizon (suffix = 1 state + n_action_steps actions)
    rms_eps: float = 1e-6
    rope_theta: float = 5_000_000.0   # Qwen3-VL default (host owns the real mRoPE via prefix)


def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device="cpu"):
    """Verbatim from lingbotvla utils."""
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


class RMSNorm(nn.Module):
    def __init__(self, hidden, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dt))


class AdaRMSNorm(nn.Module):
    """RMSNorm + FiLM (verbatim). cond = time embedding."""
    def __init__(self, hidden, cond_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = eps
        self.gamma = nn.Linear(cond_dim, hidden)
        self.beta = nn.Linear(cond_dim, hidden)

    def forward(self, hidden_states, cond):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states
        gamma = self.gamma(cond).unsqueeze(1)
        beta = self.beta(cond).unsqueeze(1)
        hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        return hidden_states.to(input_dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q [B,S,Hq,D], k [B,S,Hk,D]; cos/sin [B,S,D] -> unsqueeze head dim (dim=2)
    cos = cos.unsqueeze(2); sin = sin.unsqueeze(2)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def eager_attention(q, k, v, attn_mask):
    """our_eager_attention_forward (verbatim): q [B,S,Hq,D], k/v [B,Sk,Hk,D],
    attn_mask bool [B,S,Sk]. Manual softmax matmul (no SDPA)."""
    b, s, hq, d = q.shape
    hk = k.shape[2]
    g = hq // hk
    k = k.repeat_interleave(g, dim=2)
    v = v.repeat_interleave(g, dim=2)
    qp = q.permute(0, 2, 1, 3); kp = k.permute(0, 2, 1, 3)
    aw = torch.einsum("bhqd,bhkd->bhqk", qp, kp) * (d ** -0.5)
    big_neg = -2.3819763e38
    aw = torch.where(attn_mask[:, None, :, :], aw, big_neg)
    probs = F.softmax(aw, dim=-1).to(v.dtype)
    vp = v.permute(0, 2, 1, 3)
    out = torch.einsum("bhqk,bhkd->bhqd", probs, vp)
    return out.permute(0, 2, 1, 3).reshape(b, s, hq * d)


class DecoderLayer(nn.Module):
    def __init__(s, c: LVConfig):
        super().__init__()
        s.c = c
        qd = c.n_q_heads * c.head_dim; kd = c.n_kv_heads * c.head_dim
        s.input_layernorm = AdaRMSNorm(c.hidden, c.cond_dim, c.rms_eps)
        s.post_attention_layernorm = AdaRMSNorm(c.hidden, c.cond_dim, c.rms_eps)
        s.self_attn = nn.Module()
        s.self_attn.q_proj = nn.Linear(c.hidden, qd)
        s.self_attn.k_proj = nn.Linear(c.hidden, kd)
        s.self_attn.v_proj = nn.Linear(c.hidden, kd)
        s.self_attn.o_proj = nn.Linear(qd, c.hidden, bias=False)
        # MoE (fused 3D experts)
        s.mlp = nn.Module()
        s.mlp.gate = nn.Linear(c.hidden, c.num_experts, bias=False)
        s.mlp.register_parameter("e_score_correction_bias", nn.Parameter(torch.zeros(c.num_experts)))
        s.mlp.experts = nn.Module()
        s.mlp.experts.register_parameter("gate_proj", nn.Parameter(torch.empty(c.num_experts, c.expert_inter, c.hidden)))
        s.mlp.experts.register_parameter("up_proj", nn.Parameter(torch.empty(c.num_experts, c.expert_inter, c.hidden)))
        s.mlp.experts.register_parameter("down_proj", nn.Parameter(torch.empty(c.num_experts, c.hidden, c.expert_inter)))
        s.mlp.shared_expert = nn.Module()
        s.mlp.shared_expert.gate_proj = nn.Linear(c.hidden, c.shared_inter, bias=False)
        s.mlp.shared_expert.up_proj = nn.Linear(c.hidden, c.shared_inter, bias=False)
        s.mlp.shared_expert.down_proj = nn.Linear(c.shared_inter, c.hidden, bias=False)

    def _expert_w(s, name):
        """Return the (dequantized) 3D expert weight. int8 path: qweight[int8] * scale."""
        if getattr(s.mlp.experts, name + "_q", None) is not None:
            q = getattr(s.mlp.experts, name + "_q")
            sc = getattr(s.mlp.experts, name + "_s")
            return q.to(sc.dtype) * sc
        return getattr(s.mlp.experts, name)

    def moe(s, x):  # x [B,S,H]
        c = s.c
        b, sq, h = x.shape
        xf = x.reshape(-1, h)
        logits = F.linear(xf.float(), s.mlp.gate.weight.float())
        scores = logits.sigmoid() if c.router_activation == "sigmoid" else F.softmax(logits, dim=1, dtype=torch.float)
        choice = scores + s.mlp.e_score_correction_bias.unsqueeze(0)
        _, sel = torch.topk(choice, c.top_k, dim=-1)
        rw = scores.gather(1, sel)
        if c.norm_topk_prob:
            rw = rw / (rw.sum(-1, keepdim=True) + 1e-20)
        rw = (rw * c.routed_scaling).to(x.dtype)
        mask = F.one_hot(sel, c.num_experts).to(x.dtype)          # [T,k,E]
        weights = (mask * rw.unsqueeze(-1)).sum(1)                 # [T,E]  (dense combine, validated exact)
        g = torch.einsum("td,eid->tei", xf, s._expert_w("gate_proj"))
        u = torch.einsum("td,eid->tei", xf, s._expert_w("up_proj"))
        hexp = F.silu(g) * u                                       # [T,E,inter]
        eo = torch.einsum("tei,edi->ted", hexp, s._expert_w("down_proj"))  # [T,E,H]
        out = torch.einsum("ted,te->td", eo, weights)
        sh = F.silu(s.mlp.shared_expert.gate_proj(xf)) * s.mlp.shared_expert.up_proj(xf)
        out = out + s.mlp.shared_expert.down_proj(sh)
        return out.reshape(b, sq, h)

    def forward(s, h, cond, prefix_k, prefix_v, cos, sin, attn_mask):
        c = s.c
        normed = s.input_layernorm(h, cond)
        q = s.self_attn.q_proj(normed).view(*normed.shape[:-1], c.n_q_heads, c.head_dim)
        k = s.self_attn.k_proj(normed).view(*normed.shape[:-1], c.n_kv_heads, c.head_dim)
        v = s.self_attn.v_proj(normed).view(*normed.shape[:-1], c.n_kv_heads, c.head_dim)
        q, k = apply_rope(q, k, cos, sin)
        k = torch.cat([prefix_k, k], dim=1)
        v = torch.cat([prefix_v, v], dim=1)
        att = eager_attention(q, k, v, attn_mask)
        h = h + s.self_attn.o_proj(att)
        normed2 = s.post_attention_layernorm(h, cond)
        h = h + s.moe(normed2)
        return h


class LingbotActionExpert(nn.Module):
    def __init__(s, c: LVConfig):
        super().__init__()
        s.c = c
        s.layers = nn.ModuleList([DecoderLayer(c) for _ in range(c.num_layers)])
        s.norm = RMSNorm(c.hidden, c.rms_eps)
        s.state_proj = nn.Linear(c.state_dim, c.proj_width)
        s.action_in_proj = nn.Linear(c.action_dim, c.proj_width)
        s.action_out_proj = nn.Linear(c.proj_width, c.action_dim)
        s.action_time_mlp_in = nn.Linear(c.proj_width * 2, c.proj_width)
        s.action_time_mlp_out = nn.Linear(c.proj_width, c.proj_width)

    def quantize_experts_int8(s):
        """Per-output-channel weight-only int8 on the 3D MoE experts (the bulk of the
        params). torchao can't einsum quantized 3D tensors, so we store int8 + fp16
        scale and dequantize in _expert_w before the einsum. Halves the asset (~3.6GB
        fp16 -> ~1.9GB) so it loads on the ANE."""
        for l in s.layers:
            e = l.mlp.experts
            for name in ("gate_proj", "up_proj", "down_proj"):
                w = getattr(e, name).data.float()                 # [E, O, I]
                scale = w.abs().amax(dim=2, keepdim=True) / 127.0  # [E, O, 1]
                q = torch.round(w / scale.clamp(min=1e-12)).clamp(-127, 127).to(torch.int8)
                delattr(e, name)
                e.register_buffer(name + "_q", q)
                e.register_buffer(name + "_s", scale.half())

    def embed_suffix(s, state, noisy_actions, timestep):
        c = s.c
        state_emb = s.state_proj(state)
        time_emb = create_sinusoidal_pos_embedding(timestep, c.proj_width, 4e-3, 4.0, device=state.device).to(state.dtype)
        action_emb = s.action_in_proj(noisy_actions)
        te = time_emb[:, None, :].expand(-1, action_emb.shape[1], -1)
        at = s.action_time_mlp_out(F.silu(s.action_time_mlp_in(torch.cat([action_emb, te], dim=-1))))
        embs = torch.cat([state_emb[:, None], at], dim=1)   # [B, 1+chunk, H]
        return time_emb, embs

    def _rope_cache(s, position_ids):
        # position_ids [B, S] (text-type; mRoPE sections equal). Standard 1D RoPE.
        c = s.c
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, c.head_dim, 2, device=position_ids.device, dtype=torch.float32) / c.head_dim))
        ang = position_ids.float()[..., None] * inv[None, None, :]      # [B,S,D/2]
        emb = torch.cat([ang, ang], dim=-1)
        return emb.cos(), emb.sin()

    def forward(s, state, noisy_actions, timestep, prefix_k, prefix_v, position_ids, attn_mask):
        """Deployable action-denoise step. prefix_k/v: [num_layers, B, prefix_len, n_kv_heads, head_dim]
        (VLM cache, host-prefilled + roped). Returns velocity [B, n_action_steps, action_dim]."""
        c = s.c
        cond, h = s.embed_suffix(state, noisy_actions, timestep)
        cos, sin = s._rope_cache(position_ids)
        for i, layer in enumerate(s.layers):
            h = layer(h, cond, prefix_k[i], prefix_v[i], cos, sin, attn_mask)
        h = s.norm(h)
        return s.action_out_proj(h[:, -c.n_action_steps:])
