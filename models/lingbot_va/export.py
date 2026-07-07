# /// script
# requires-python = ">=3.12"
# ///
"""LingBot-VA action-denoise-step export lane (LeRobot v0.6.0 `lingbot_va`).

LingBot-VA is an autoregressive video-action world model on the Wan2.2 stack
(WanTransformer3DModel dual-stream video+action DiT + Wan VAE + UMT5). This lane
exports the deployable ACTION core — the cache-free action-stream velocity graph:

  action_denoise_step(noisy_action_latents, text_emb, grid_id, timesteps) -> velocity

reached via `WanTransformer3DModel.forward(input_dict, action_mode=True,
update_cache=0)`. The host owns the Wan VAE (frame encode), the UMT5 text encoder,
the CFG + Euler action scheduler, and the streaming video-KV cache (this graph is
the cache-free single-chunk action denoiser — the video-KV-as-graph-I/O path is a
follow-up for full streaming deployment).

TWO rewrites make it lower on coreai_torch 0.4.1 (op-coverage PROVEN):
  1. attn_mode="torch" -> WanAttention.attn_op = custom_sdpa (plain SDPA; flex_attention
     is training-only) — no change needed.
  2. RoPE: the upstream apply_rotary_emb uses torch.view_as_complex/view_as_real, and
     coreai_torch has NO complex dtype (KeyError: torch.complex128). We monkeypatch
     WanRotaryPosEmbed to emit real (cos, sin) and WanAttention.forward to apply the
     mathematically-identical real rotation (verified max abs diff 1.7e-6).

Usage:
  .venv/bin/python models/lingbot_va/export.py export \
      --weights build/_lingbot_va/model.safetensors --out build/lingbot-va-base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

LINGBOT_SRC = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric/build/_src_lingbot_va")

# config.json (lerobot/lingbot_va_base)
PATCH_SIZE = (1, 2, 2)
NUM_HEADS = 24
HEAD_DIM = 128
IN_CH = OUT_CH = 48
ACTION_DIM = 30
TEXT_DIM = 4096
FREQ_DIM = 256
FFN_DIM = 14336
NUM_LAYERS = 30
ROPE_MAX = 1024
ACTION_PER_FRAME = 16
FRAME_CHUNK = 2          # F latent frames per action chunk
CTX_TEXT_LEN = 32        # synthetic UMT5 context length (host supplies the real one)


def _install_real_rope(utils):
    """Replace complex-number RoPE with a real (cos,sin) rewrite that lowers.

    Upstream WanRotaryPosEmbed builds 3D (frame/height/width) angles then
    `torch.polar` -> complex freqs_cis; WanAttention.apply_rotary_emb then does a
    complex multiply. coreai_torch has no complex dtype, so we emit real (cos,sin)
    and apply the mathematically-identical real rotation.
    """

    def rope_forward_real(self, grid_ids):  # grid_ids [B, 3, L] -> [B, L, D/2, 2]
        f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(grid_ids.device)
        h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(grid_ids.device)
        w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(grid_ids.device)
        freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()  # [B, L, D/2]
        return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)  # [B, L, D/2, 2]

    utils.WanRotaryPosEmbed.forward = rope_forward_real

    # Real-valued apply_rotary_emb inside a cache-free WanAttention.forward.
    # The model does `rope(grid)[:, :, None]` -> rotary_emb is [B, L, 1, D/2, 2],
    # so cos/sin are [B, L, 1, D/2] and broadcast over the heads dim of the query.
    def attn_forward_real(self, q, k, v, rotary_emb, update_cache=0, cache_name="pos"):
        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query).unflatten(2, (self.heads, -1))
        key = self.norm_k(key).unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:
            cos = rotary_emb[..., 0]      # [B, L, 1, D/2]
            sin = rotary_emb[..., 1]

            def rope(x):                  # x [B, L, heads, head_dim]
                xp = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
                xe, xo = xp[..., 0], xp[..., 1]
                oe = xe * cos - xo * sin
                oo = xe * sin + xo * cos
                return torch.stack([oe, oo], dim=-1).flatten(3).to(x.dtype)

            query = rope(query)
            key = rope(key)
        # cache-free: attend within the current tokens (host owns the video-KV cache)
        hidden = self.attn_op(query, key, value)
        hidden = hidden.flatten(2, 3).type_as(query)
        hidden = self.to_out[0](hidden)
        hidden = self.to_out[1](hidden)
        return hidden

    utils.WanAttention.forward = attn_forward_real


def _stub_lerobot():
    """utils.py only needs `lerobot.utils.import_utils`'s availability flags at
    import time (diffusers/transformers are actually present in fabric .venv) — stub
    it so we don't install the whole lerobot stack into the coreai_torch env."""
    import types
    for name in ("lerobot", "lerobot.utils"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    iu = types.ModuleType("lerobot.utils.import_utils")
    iu._diffusers_available = True
    iu._transformers_available = True
    sys.modules["lerobot.utils.import_utils"] = iu


def _build_transformer():
    sys.path.insert(0, str(LINGBOT_SRC))
    _stub_lerobot()
    import utils as lb  # diffusers-based

    _install_real_rope(lb)
    model = lb.WanTransformer3DModel(
        patch_size=PATCH_SIZE, num_attention_heads=NUM_HEADS, attention_head_dim=HEAD_DIM,
        in_channels=IN_CH, out_channels=OUT_CH, action_dim=ACTION_DIM, text_dim=TEXT_DIM,
        freq_dim=FREQ_DIM, ffn_dim=FFN_DIM, num_layers=NUM_LAYERS, cross_attn_norm=True,
        eps=1e-6, rope_max_seq_len=ROPE_MAX, attn_mode="torch",
    )
    return model.eval()


def _load_transformer_weights(model, weights: Path) -> None:
    from safetensors import safe_open

    prefix = "transformer."
    subset = {}
    with safe_open(str(weights), framework="pt", device="cpu") as h:
        for key in h.keys():
            if key.startswith(prefix):
                subset[key[len(prefix):]] = h.get_tensor(key)
    if not subset:
        raise SystemExit(f"no transformer weights with prefix {prefix!r}")
    missing, unexpected = model.load_state_dict(subset, strict=False)
    # attn_caches / rope buffers are non-persistent; ignore.
    unexpected = [k for k in unexpected if "attn_caches" not in k]
    if unexpected:
        print(f"warning: {len(unexpected)} unexpected key(s); first: {unexpected[:6]}")
    real_missing = [k for k in missing if "attn_cache" not in k and not k.endswith(".freqs")]
    if real_missing:
        print(f"warning: {len(real_missing)} missing key(s); first: {real_missing[:6]}")


class ActionDenoiseStep(torch.nn.Module):
    """Cache-free action-stream velocity step (action_mode=True, update_cache=0)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, noisy_action_latents, text_emb, grid_id, timesteps):
        input_dict = {
            "noisy_latents": noisy_action_latents,   # [B, action_dim, F, apf, 1]
            "text_emb": text_emb,                    # [B, L2, text_dim]
            "grid_id": grid_id,                      # [1, L1]
            "timesteps": timesteps,                  # [B, 1]
        }
        return self.model(input_dict, update_cache=0, action_mode=True, train_mode=False)


def _example_inputs():
    b = 1
    L1 = FRAME_CHUNK * ACTION_PER_FRAME * 1          # (f h w) for action = 2*16*1 = 32
    noisy = torch.zeros(b, ACTION_DIM, FRAME_CHUNK, ACTION_PER_FRAME, 1)
    text = torch.zeros(b, CTX_TEXT_LEN, TEXT_DIM)
    grid = torch.zeros(1, 3, L1, dtype=torch.long)   # 3 axes (f, h, w) per token
    ts = torch.zeros(b, FRAME_CHUNK, dtype=torch.long)  # one timestep per latent frame; repeat_interleave(h*w) -> L1
    return noisy, text, grid, ts


def cmd_export(args) -> None:
    from coreai_torch import TorchConverter, get_decomp_table

    model = _build_transformer()
    _load_transformer_weights(model, args.weights)
    wrapper = ActionDenoiseStep(model).eval()

    with torch.no_grad():
        ep = torch.export.export(wrapper, args=_example_inputs(), strict=False)
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(
        ep, input_names=["noisy_action_latents", "text_emb", "grid_id", "timesteps"],
        output_names=["velocity"], entrypoint_name="action_denoise_step")
    prog = conv.to_coreai()
    prog.optimize()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    aimodel = out / f"{out.name}.aimodel"
    prog.save_asset(aimodel)
    contract = {
        "entrypoint": "action_denoise_step", "action_dim": ACTION_DIM, "frame_chunk": FRAME_CHUNK,
        "action_per_frame": ACTION_PER_FRAME, "text_dim": TEXT_DIM, "ctx_text_len": CTX_TEXT_LEN,
        "host_components": ["wan_vae", "umt5_text_encoder", "cfg_action_scheduler",
                            "streaming_video_kv_cache", "un_normalize"],
        "note": "cache-free single-chunk action denoiser; video-KV-as-I/O is a follow-up",
    }
    (out / "lingbot-va-action-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"ok: lowered LingBot-VA action head -> {aimodel}")


def main() -> None:
    ap = argparse.ArgumentParser(description="LingBot-VA action-head export")
    ap.add_argument("phase", nargs="?", default="export", choices=["export"])
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cmd_export(args)


if __name__ == "__main__":
    main()
