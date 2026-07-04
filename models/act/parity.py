# /// script
# requires-python = ">=3.12"
# ///
"""action_parity Gate B for a single-graph ACT policy .aimodel (robot-free).

ACT is DETERMINISTIC (inference sets the VAE latent to its mean) and single-graph,
so parity = drive the SAME observations through (a) the torch reference policy's
`predict_action_chunk` — exactly the function that was exported — and (b) the lowered
`.aimodel`, then compare the predicted action chunks. This certifies the export is
NUMERICALLY FAITHFUL to the source policy; it does NOT certify real-world task success
(the card says so). Mirrors greedy_parity's honesty: fixed-seed inputs, min-cosine as
the headline, per-dim MAE, and a bootstrap CI so a small sample can't claim "lossless".

TWO-VENV (like export.py): `reference` runs in venv-A (.venv-lerobot, torch 2.9 +
lerobot[pi]) and writes the reference chunks to an .npz; `--compare` runs in venv-B
(fabric .venv, coreai.runtime) and drives the asset. The observations cross as plain
numpy, so no torch/version coupling.

Usage:
  .venv-lerobot/bin/python models/act/parity.py reference --repo <hf_repo> --out build/<id>
  .venv/bin/python         models/act/parity.py --compare  --out build/<id> \
        --bundle build/<id>/<id>.aimodel
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REF_NPZ = "act_parity_ref.npz"


def cmd_reference(repo: str, out: Path, n_obs: int, seed: int):
    """venv-A: torch `predict_action_chunk` on fixed-seed observations -> ref chunks."""
    import numpy as np
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained(repo).to("cpu").eval()
    cfg = policy.config
    # Infer the contract from the config so this harness generalizes across ACT
    # checkpoints (aloha/koch/so100 differ in cameras + state/action dims).
    img_key = next(k for k, f in cfg.input_features.items() if f.type == "VISUAL")
    img_shape = tuple(cfg.input_features[img_key].shape)           # e.g. (3, 480, 640)
    state_dim = int(cfg.input_features["observation.state"].shape[0])

    images, states, refs = [], [], []
    for i in range(n_obs):
        gi = torch.Generator().manual_seed(seed + i)
        img = torch.rand((1, *img_shape), generator=gi)           # pixels in [0, 1]
        st = torch.randn((1, state_dim), generator=gi) * 0.1      # small joint values
        with torch.no_grad():
            act = policy.predict_action_chunk({img_key: img, "observation.state": st})
        images.append(img.numpy())
        states.append(st.numpy())
        refs.append(act.numpy())

    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / REF_NPZ,
        images=np.concatenate(images).astype(np.float32),
        states=np.concatenate(states).astype(np.float32),
        refs=np.concatenate(refs).astype(np.float32),
        img_key=np.array(img_key),
    )
    print(f"ok: wrote {out/REF_NPZ}  (n_obs={n_obs}, img={img_shape}, state={state_dim}, "
          f"chunk={refs[0].shape[1]}x{refs[0].shape[2]})")
    print("next (venv-B): .venv/bin/python models/act/parity.py --compare --out", out,
          "--bundle", out / f"{out.name}.aimodel")


def cmd_compare(out: Path, bundle: Path):
    """venv-B: drive the asset on the SAME observations, compare vs the reference."""
    import asyncio

    import numpy as np
    from coreai.runtime import AIModel, NDArray

    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _bootstrap_ci, _environment

    d = np.load(out / REF_NPZ, allow_pickle=True)
    images, states, refs = d["images"], d["states"], d["refs"]     # [N,C,H,W] [N,S] [N,T,A]

    async def _run() -> dict:
        model = await AIModel.load(str(bundle))
        fn = model.load_function("main")
        desc = fn.desc
        # Map input names by rank (image is 4-D, state is 2-D) so we don't hardcode
        # the torch.export arg names.
        in_names = list(desc.input_names)
        rank = {n: len(desc.input_descriptor(n).shape) for n in in_names}
        img_name = next(n for n in in_names if rank[n] == 4)
        state_name = next(n for n in in_names if rank[n] == 2)
        out_name = "action_chunk" if "action_chunk" in desc.output_names else desc.output_names[0]

        cosines: list[float] = []
        abs_err = None  # accumulate |ref-got| summed over rows, per (t? no) -> per action dim
        n_rows = 0
        for i in range(len(images)):
            img = images[i:i + 1].astype(np.float32)
            st = states[i:i + 1].astype(np.float32)
            res = await fn(inputs={img_name: NDArray(img), state_name: NDArray(st)})
            got = res[out_name].numpy()[0].astype(np.float64)      # [T, A]
            ref = refs[i].astype(np.float64)                       # [T, A]
            ae = np.abs(ref - got)
            abs_err = ae.sum(axis=0) if abs_err is None else abs_err + ae.sum(axis=0)
            n_rows += ref.shape[0]
            for r in range(ref.shape[0]):
                a, b = ref[r], got[r]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na == 0.0 or nb == 0.0:
                    continue
                cosines.append(float(np.dot(a, b) / (na * nb)))

        cos = np.asarray(cosines)
        per_dim_mae = (abs_err / n_rows)                           # [A]
        lo, hi = _bootstrap_ci(cos.tolist())
        return {
            "metric": "action_parity",
            "value": float(cos.min()),
            "min_action_cosine": float(cos.min()),
            "mean_action_cosine": float(cos.mean()),
            "cosine_ci95": [lo, hi],
            "max_per_dim_mae": float(per_dim_mae.max()),
            "mean_per_dim_mae": float(per_dim_mae.mean()),
            "n_obs": int(len(images)),
            "n_action_rows": int(n_rows),
            "action_dim": int(refs.shape[2]),
            "chunk_size": int(refs.shape[1]),
            "sampler": "act",
            "deterministic": True,
            "reference": "torch predict_action_chunk (fp32 as loaded by lerobot)",
            "runner": f"coreai-fabric-parity-runner/{__version__}",
            "environment": _environment(),
        }

    result = asyncio.run(_run())
    # Drop the protocol JSON next to the bundle so `coreai-fabric verify` can
    # RECORD this on-hardware measurement (it recomputes pass/fail from `value`
    # vs the recipe threshold — it never trusts a self-reported status). This is
    # the documented handoff for the intrinsically two-venv action lane.
    emit = out / "action-parity-measured.json"
    emit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nok: wrote {emit} (feeds `coreai-fabric verify {out.name}`)")


def main():
    ap = argparse.ArgumentParser(description="ACT action_parity (see docs/vla-export-runbook.md)")
    ap.add_argument("phase", nargs="?", default="reference", choices=["reference"])
    ap.add_argument("--repo", default="lerobot/act_aloha_sim_transfer_cube_human")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--n-obs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.compare:
        cmd_compare(args.out, args.bundle or (args.out / f"{args.out.name}.aimodel"))
    else:
        cmd_reference(args.repo, args.out, args.n_obs, args.seed)


if __name__ == "__main__":
    main()
