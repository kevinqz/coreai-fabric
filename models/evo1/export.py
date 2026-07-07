# /// script
# requires-python = ">=3.12"
# ///
"""EVO1 action-head export lane (LeRobot v0.6.0 `evo1` policy).

EVO1 = InternVL3-1B VL embedder + a flow-matching action head, the same split
discipline as VLA-JEPA / pi0. This lane exports the deployable core — the
per-step velocity graph:

  action_denoise_step(fused_tokens, x_t, time_index, state, embodiment_id) -> velocity

The host owns the InternVL3 VL embedding (context) + the Euler loop
(num_inference_timesteps) + un-normalization. Gate B is action_parity
(models/evo1/parity.py), fixed synthetic context + fixed noise.

`flow_matching.py` in the lerobot evo1 package is pure torch, so we import it
standalone (no lerobot/transformers) and load only `model.action_head.*` from the
checkpoint — single env (.venv, coreai_torch). Op-coverage proven on
coreai_torch 0.4.1 / coremltools 9.0.

Usage:
  .venv/bin/python models/evo1/export.py export \
      --weights build/_evo1/model.safetensors --out build/evo1-so100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# The lerobot evo1 package dir (flow_matching.py is pure-torch + self-contained).
EVO1_PKG = Path(
    "/Users/kevinsaltarelli/Dev/Github/coreai-fabric/.venv-lerobot/lib/python3.12/"
    "site-packages/lerobot/policies/evo1"
)

# Evo1Config dims (verified against the checkpoint: embed_dim = InternVL3-1B hidden).
# Dims verified against the Beilinghamburger/evo1_so100_vla checkpoint tensors
# (model.action_head.*): action_encoder.W1 [896,7], pos_encoding.pe [1,16,896],
# state_encoder.fc1 [1024,7], mlp_head.fc2 [112,1024] (= 16*7). The SO-100 arm is
# 7-DoF and this policy uses a chunk_size of 16 — NOT the generic Evo1Config
# defaults (chunk 50 / dim 24) used in the random-init op-coverage probe.
EMBED_DIM = 896
HIDDEN_DIM = 1024
NUM_HEADS = 8
NUM_LAYERS = 8
NUM_CATEGORIES = 1
HORIZON = 16            # config.chunk_size (pos_encoding.pe length)
PER_ACTION_DIM = 7      # config.max_action_dim (SO-100 = 6 joints + gripper)
STATE_DIM = 7           # config.max_state_dim
CTX_TOKENS = 16         # synthetic VL context length (host supplies the real one)


def _build_head():
    sys.path.insert(0, str(EVO1_PKG))
    import flow_matching as fm  # pure torch

    head = fm.FlowmatchingActionHead(
        embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, action_dim=HORIZON * PER_ACTION_DIM,
        horizon=HORIZON, per_action_dim=PER_ACTION_DIM, num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS, dropout=0.0, num_inference_timesteps=32,
        num_categories=NUM_CATEGORIES, state_dim=STATE_DIM, state_hidden_dim=HIDDEN_DIM,
    )
    return head.eval()


def _load_action_head_weights(head, weights: Path | None) -> None:
    if weights is None:
        return
    from safetensors import safe_open

    prefix = "model.action_head."
    subset = {}
    with safe_open(str(weights), framework="pt", device="cpu") as h:
        for key in h.keys():
            if key.startswith(prefix):
                subset[key[len(prefix):]] = h.get_tensor(key)
    if not subset:
        raise SystemExit(f"no action-head weights found with prefix {prefix!r}")
    missing, unexpected = head.load_state_dict(subset, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected action-head keys: {unexpected[:8]}")
    if missing:
        print(f"warning: {len(missing)} missing action-head key(s); first: {missing[:6]}")


class DenoiseStep(torch.nn.Module):
    """One flow-matching velocity step (predict_velocity from get_action)."""

    def __init__(self, h):
        super().__init__()
        self.h = h

    def forward(self, fused_tokens, x_t, time_index, state, embodiment_id):
        h = self.h
        context_tokens, kpm, emb = h._prepare_context(fused_tokens, state, embodiment_id, None)
        if kpm is None:
            # MPSGraph WORKAROUND (macOS 27 / Xcode 27-beta): loading an .aimodel
            # whose cross-attention has NO key_padding_mask SEGFAULTs inside
            # MetalPerformanceShadersGraph's FoldMultiplyIntoSDPAScale pass (it folds
            # the preceding LayerNorm gamma into the reconstructed SDPA scale and
            # crashes). Ablation: the SAME graph WITH a key_padding_mask loads fine.
            # An all-False mask masks nothing (numerically identical) and forces the
            # masked-SDPA codepath — BUT a constant all-False mask gets folded back to
            # "no mask" and re-crashes, so it must be DATA-DEPENDENT: `sum(|ctx|) < -1`
            # is always False (|ctx| >= 0) yet MPSGraph can't prove it constant, so the
            # masked path survives optimization. Both facts verified by ablation probes.
            kpm = context_tokens.abs().sum(dim=-1) < -1.0
        time_emb = h.time_pos_enc(1000)[:, time_index, :].squeeze(0).to(dtype=context_tokens.dtype)
        action_tokens = h._project_actions(x_t, emb).to(dtype=context_tokens.dtype)
        x = action_tokens
        for block in h.transformer_blocks:
            x = block(x, context_tokens, time_emb, kpm)
        x = h.norm_out(x)
        x_pooled = h.seq_pool_proj(x.reshape(x.shape[0], -1)) if h.horizon > 1 else x.squeeze(1)
        return h.mlp_head(x_pooled, emb)


def cmd_export(args) -> None:
    from coreai_torch import TorchConverter, get_decomp_table

    head = _build_head()
    _load_action_head_weights(head, args.weights)
    wrapper = DenoiseStep(head).eval()

    b = 1
    fused = torch.zeros(b, CTX_TOKENS, EMBED_DIM)
    x_t = torch.zeros(b, HORIZON, PER_ACTION_DIM)
    time_index = torch.zeros(b, dtype=torch.long)
    state = torch.zeros(b, STATE_DIM)
    emb_id = torch.zeros(b, dtype=torch.long)

    with torch.no_grad():
        ep = torch.export.export(
            wrapper, args=(fused, x_t, time_index, state, emb_id), strict=False)
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(
        ep,
        input_names=["fused_tokens", "x_t", "time_index", "state", "embodiment_id"],
        output_names=["velocity"], entrypoint_name="action_denoise_step")
    prog = conv.to_coreai()
    prog.optimize()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    aimodel = out / f"{out.name}.aimodel"
    prog.save_asset(aimodel)
    contract = {
        "entrypoint": "action_denoise_step",
        "embed_dim": EMBED_DIM, "horizon": HORIZON, "per_action_dim": PER_ACTION_DIM,
        "state_dim": STATE_DIM, "num_categories": NUM_CATEGORIES, "ctx_tokens": CTX_TOKENS,
        "host_components": ["internvl3_embedder", "euler_loop", "un_normalize"],
    }
    (out / "evo1-action-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"ok: lowered EVO1 action head -> {aimodel}")
    print(f"ok: wrote {out / 'evo1-action-contract.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="EVO1 action-head export")
    ap.add_argument("phase", nargs="?", default="export", choices=["export"])
    ap.add_argument("--weights", type=Path, help="evo1 checkpoint model.safetensors (loads model.action_head.*)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cmd_export(args)


if __name__ == "__main__":
    main()
