# /// script
# requires-python = ">=3.12"
# ///
"""Robometer-4B reward-head export lane (LeRobot v0.6.0 `rewards/robometer`).

Robometer = a Qwen3-VL-4B-Instruct backbone + small MLP reward heads (progress +
success), the SAME split discipline as EVO1 / VLA-JEPA / pi0. This lane exports the
deployable task-specific core — the reward heads:

  reward_heads(frame_embeddings) -> (progress_logits, success_logits)

where `frame_embeddings` (B, T, hidden) are the per-frame `<|prog_token|>` hidden
states the host reads out of Qwen3-VL. The host owns the Qwen3-VL-4B backbone (a
standard VLM), the prog-token extraction, and the decode (softmax-weighted bin mean
clamped to [0,1] for progress; sigmoid for success). Gate B is graph_output_cosine
(models/robometer/parity.py), the non-autoregressive encoder lane.

`RobometerPredictionHead` is pure torch (nn.Sequential of Linear/LayerNorm/GELU) so
we rebuild it standalone from the checkpoint's `progress_head.*` / `success_head.*`
tensors (dims inferred from the tensor shapes — no lerobot import needed) and lower
on coreai_torch 0.4.1 / coremltools 9.0.

Usage:
  .venv/bin/python models/robometer/export.py export \
      --weights build/_robometer/model.safetensors --out build/robometer-4b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


def _prediction_head(hidden_dim: int, output_size: int) -> nn.Sequential:
    """Rebuild RobometerPredictionHead WITHOUT the sigmoid (raw logits out).

    Upstream applies sigmoid only on the continuous-progress path; Robometer-4B
    uses discrete progress (bins) + raw success logits (sigmoid at decode). We ship
    raw logits either way — the host decodes — so no trailing activation here.
    """
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.LayerNorm(hidden_dim // 2),
        nn.GELU(),
        nn.Identity(),  # placeholder for the training-time Dropout (index parity)
        nn.Linear(hidden_dim // 2, output_size),
    )


def _load_head(head: nn.Sequential, weights: Path, prefix: str) -> None:
    from safetensors import safe_open

    subset = {}
    with safe_open(str(weights), framework="pt", device="cpu") as h:
        for key in h.keys():
            if key.startswith(prefix):
                subset[key[len(prefix):]] = h.get_tensor(key)
    if not subset:
        raise SystemExit(f"no head weights with prefix {prefix!r}")
    missing, unexpected = head.load_state_dict(subset, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected {prefix} keys: {unexpected[:8]}")
    if missing:
        # Identity/GELU carry no params — only Dropout's absence is expected.
        print(f"note: {len(missing)} missing {prefix} key(s) (expected: none with params)")


def _infer_dims(weights: Path) -> tuple[int, int]:
    """(hidden_dim, progress_bins) from the checkpoint head tensors."""
    from safetensors import safe_open

    hidden_dim = progress_bins = None
    with safe_open(str(weights), framework="pt", device="cpu") as h:
        keys = list(h.keys())
        for k in keys:
            if k.startswith("progress_head.0.") and k.endswith("weight"):
                hidden_dim = h.get_slice(k).get_shape()[1]           # Linear in_features
            if k.startswith("progress_head.4.") and k.endswith("weight"):
                progress_bins = h.get_slice(k).get_shape()[0]        # final Linear out
    if hidden_dim is None or progress_bins is None:
        raise SystemExit("could not infer head dims (progress_head.0/.4 not found)")
    return int(hidden_dim), int(progress_bins)


class RewardHeads(nn.Module):
    """Per-frame Robometer reward core: frame_embeddings -> (progress, success)."""

    def __init__(self, hidden_dim: int, progress_bins: int):
        super().__init__()
        self.progress_head = _prediction_head(hidden_dim, progress_bins)
        self.success_head = _prediction_head(hidden_dim, 1)

    def forward(self, frame_embeddings):
        progress_logits = self.progress_head(frame_embeddings)          # (B, T, bins)
        success_logits = self.success_head(frame_embeddings).squeeze(-1)  # (B, T)
        return progress_logits, success_logits


def build(weights: Path):
    hidden_dim, bins = _infer_dims(weights)
    heads = RewardHeads(hidden_dim, bins).eval()
    _load_head(heads.progress_head, weights, "progress_head.")
    _load_head(heads.success_head, weights, "success_head.")
    return heads, hidden_dim, bins


# Fixed export shapes: heads are pointwise over the hidden dim, so T is a free
# batch axis — a static (1, MAX_FRAMES, hidden) graph covers any frame count the
# host feeds (it can call per-frame or in a T-chunk).
MAX_FRAMES = 8


def cmd_export(args) -> None:
    from coreai_torch import TorchConverter, get_decomp_table

    heads, hidden_dim, bins = build(args.weights)
    b, t = 1, MAX_FRAMES
    frame_embeddings = torch.zeros(b, t, hidden_dim)

    with torch.no_grad():
        ep = torch.export.export(heads, args=(frame_embeddings,), strict=False)
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(
        ep, input_names=["frame_embeddings"],
        output_names=["progress_logits", "success_logits"],
        entrypoint_name="reward_heads")
    prog = conv.to_coreai()
    prog.optimize()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    aimodel = out / f"{out.name}.aimodel"
    prog.save_asset(aimodel)
    contract = {
        "entrypoint": "reward_heads", "hidden_dim": hidden_dim, "progress_bins": bins,
        "max_frames": MAX_FRAMES, "outputs": ["progress_logits", "success_logits"],
        "host_components": ["qwen3_vl_4b_backbone", "prog_token_extraction",
                            "decode (progress bin-mean clamp[0,1]; success sigmoid)"],
    }
    (out / "robometer-reward-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"ok: lowered Robometer reward heads (hidden={hidden_dim}, bins={bins}) -> {aimodel}")
    print(f"ok: wrote {out / 'robometer-reward-contract.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Robometer reward-head export")
    ap.add_argument("phase", nargs="?", default="export", choices=["export"])
    ap.add_argument("--weights", type=Path, required=True,
                    help="Robometer model.safetensors (loads progress_head.*/success_head.*)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cmd_export(args)


if __name__ == "__main__":
    main()
