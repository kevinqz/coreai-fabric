# /// script
# requires-python = ">=3.12"
# ///
"""ACT (Action Chunking Transformer) policy -> Apple .aimodel, single-graph.

Per-model export script (fabric prints, does not drive). ACT is DETERMINISTIC and
single-graph (no VLM, no flow-matching sampler) — the simplest VLA/action lane
target, used to prove the pipeline before pi0. See docs/vla-export-runbook.md Phase 1.

TWO-VENV (like pi0): export in venv-A (.venv-lerobot, torch 2.9 + lerobot[pi]) ->
policy.pt2; --lower in venv-B (fabric .venv, torch 2.9 + coreai_torch) -> the .aimodel.

Usage:
  .venv-lerobot/bin/python models/act/export.py export --repo lerobot/act_aloha_sim_transfer_cube_human --out build/act-aloha-cube
  .venv/bin/python         models/act/export.py --lower --out build/act-aloha-cube
"""
from __future__ import annotations

import argparse
from pathlib import Path

# ALOHA transfer-cube: 1 cam 480x640 + state(14) -> action chunk [100, 14] (VERIFIED)
IMG_SHAPE, STATE_DIM = (1, 3, 480, 640), 14


class ACTWrap:  # defined as a plain factory to avoid importing torch at module load (venv-B)
    pass


def _build(repo: str):
    """venv-A: load the ACT policy on CPU + wrap (image,state)->action_chunk (the traced graph).
    Config-DRIVEN (like parity.py): infers the camera key + image/state shapes from the checkpoint,
    so it generalizes across ACT checkpoints (aloha=top/480x640/state14, so-arm101=wrist/480x640/
    state6, ...). Returns (wrapper, img_shape, state_dim) so cmd_export sizes the dummy inputs.
    Single-camera ACT only (the common case); multi-cam checkpoints need a wider wrapper."""
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(repo).to("cpu").eval()
    cfg = policy.config
    vis = [(k, f) for k, f in cfg.input_features.items() if str(getattr(f, "type", "")).endswith("VISUAL")]
    if len(vis) != 1:
        raise SystemExit(f"ACT export currently supports single-camera checkpoints; found {len(vis)} "
                         f"VISUAL inputs {[k for k, _ in vis]}. Extend _build for multi-cam.")
    img_key, img_feat = vis[0]
    img_shape = (1, *tuple(img_feat.shape))                     # e.g. (1,3,480,640)
    state_dim = int(cfg.input_features["observation.state"].shape[0])

    class _W(torch.nn.Module):
        def __init__(self, p): super().__init__(); self.p = p
        def forward(self, image, state):
            return self.p.predict_action_chunk({img_key: image, "observation.state": state})
    return _W(policy).eval(), img_shape, state_dim


def cmd_export(repo: str, out: Path):
    import torch
    w, img_shape, state_dim = _build(repo)
    img, st = torch.zeros(*img_shape), torch.zeros(1, state_dim)
    ep = torch.export.export(w, args=(img, st), strict=False)
    out.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, str(out / "policy.pt2"))
    print(f"ok: wrote {out}/policy.pt2")
    print("next (venv-B): .venv/bin/python models/act/export.py --lower --out", out)


def cmd_lower(out: Path):
    """venv-B: load policy.pt2 -> coreai_torch lower -> policy/main.aimodel (mirrors llm_export.py:156-170)."""
    import torch
    from coreai_torch import TorchConverter, get_decomp_table
    ep = torch.export.load(str(out / "policy.pt2"))
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter().add_exported_program(ep, output_names=["action_chunk"])
    prog = conv.to_coreai()
    prog.optimize()
    out.mkdir(parents=True, exist_ok=True)
    aimodel = out / f"{out.name}.aimodel"          # standard single-bundle layout
    prog.save_asset(aimodel)                        # -> <name>.aimodel/{main.mlirb,main.hash,metadata.json}
    print(f"ok: lowered ACT policy -> {aimodel} (single deterministic graph)")


def main():
    ap = argparse.ArgumentParser(description="ACT single-graph export (see docs/vla-export-runbook.md)")
    ap.add_argument("phase", nargs="?", default="export", choices=["export"])
    ap.add_argument("--repo", default="lerobot/act_aloha_sim_transfer_cube_human")
    ap.add_argument("--lower", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cmd_lower(args.out) if args.lower else cmd_export(args.repo, args.out)


if __name__ == "__main__":
    main()
