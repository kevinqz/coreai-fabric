# /// script
# requires-python = ">=3.12"
# ///
"""action_parity Gate B for the single-graph GREEDY VQ-BeT .aimodel (robot-free).

VQ-BeT rollout is one forward pass (like ACT), but its code selection is stochastic
(torch.multinomial). We converted the GREEDY variant (multinomial -> argmax); parity
therefore drives BOTH sides deterministically: the torch reference runs
`vqbet(batch, rollout=True)` under the SAME multinomial->argmax patch the export used,
and the asset has argmax baked in. Same fixed-seed stacked observations both sides ->
compare the action chunk (min chunk-cosine + per-dim MAE + bootstrap CI). Certifies the
GREEDY export is numerically faithful to the GREEDY source policy — NOT the stochastic
sampler, NOT task success (the card says so).

TWO-VENV: `reference` (venv-A) writes chunks; `--compare` (venv-B) drives the asset.

Usage:
  .venv-lerobot/bin/python models/vqbet/parity.py reference --repo <repo> --out build/vqbet-pusht
  .venv/bin/python         models/vqbet/parity.py --compare  --out build/vqbet-pusht
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export import OBS_IMAGES, OBS_STATE, greedy_codes  # noqa: E402

REF_NPZ = "vqbet_parity_ref.npz"


def cmd_reference(repo: str, out: Path, n_frames: int, seed: int):
    """venv-A: greedy vqbet rollout on fixed-seed stacked observations -> ref chunks."""
    import numpy as np
    import torch
    from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

    policy = VQBeTPolicy.from_pretrained(repo).to("cpu").eval()
    cfg = policy.config
    n_obs = int(cfg.n_obs_steps)
    img_key = next(k for k, f in cfg.input_features.items() if f.type == "VISUAL")
    C, H, W = tuple(cfg.input_features[img_key].shape)
    n_cam = sum(1 for f in cfg.input_features.values() if f.type == "VISUAL")
    state_dim = int(cfg.input_features[OBS_STATE].shape[0])

    images, states, refs = [], [], []
    with greedy_codes(), torch.no_grad():
        for i in range(n_frames):
            gi = torch.Generator().manual_seed(seed + i)
            img = torch.rand(1, n_obs, n_cam, C, H, W, generator=gi)
            st = torch.randn(1, n_obs, state_dim, generator=gi) * 0.1
            act = policy.vqbet({OBS_IMAGES: img, OBS_STATE: st}, rollout=True)
            images.append(img.numpy()); states.append(st.numpy()); refs.append(act.numpy())

    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / REF_NPZ,
             images=np.concatenate(images).astype(np.float32),
             states=np.concatenate(states).astype(np.float32),
             refs=np.concatenate(refs).astype(np.float32))
    print(f"ok: wrote {out/REF_NPZ}  (n_frames={n_frames}, n_obs={n_obs}, "
          f"chunk={refs[0].shape[1]}x{refs[0].shape[2]}, GREEDY)")
    print("next (venv-B): .venv/bin/python models/vqbet/parity.py --compare --out", out)


def cmd_compare(out: Path, bundle: Path):
    """venv-B: drive the asset on the SAME observations, compare vs the greedy reference."""
    import asyncio

    import numpy as np
    from coreai.runtime import AIModel, NDArray

    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _bootstrap_ci, _environment

    d = np.load(out / REF_NPZ, allow_pickle=True)
    images, states, refs = d["images"], d["states"], d["refs"]     # [N,n_obs,cam,C,H,W] [N,n_obs,S] [N,T,A]

    async def _run() -> dict:
        model = await AIModel.load(str(bundle))
        fn = model.load_function("main")
        desc = fn.desc
        in_names = list(desc.input_names)
        rank = {n: len(desc.input_descriptor(n).shape) for n in in_names}
        img_name = max(in_names, key=lambda n: rank[n])          # images is the 6-D input
        state_name = min(in_names, key=lambda n: rank[n])        # state is the 3-D input
        out_name = "action_chunk" if "action_chunk" in desc.output_names else desc.output_names[0]

        cosines, abs_err, n_rows = [], None, 0
        for i in range(len(images)):
            res = await fn(inputs={img_name: NDArray(images[i:i + 1].astype(np.float32)),
                                   state_name: NDArray(states[i:i + 1].astype(np.float32))})
            got = res[out_name].numpy()[0].astype(np.float64)     # [T, A]
            ref = refs[i].astype(np.float64)
            ae = np.abs(ref - got)
            abs_err = ae.sum(axis=0) if abs_err is None else abs_err + ae.sum(axis=0)
            n_rows += ref.shape[0]
            for r in range(ref.shape[0]):
                a, b = ref[r], got[r]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na and nb:
                    cosines.append(float(np.dot(a, b) / (na * nb)))
        cos = np.asarray(cosines)
        per_dim = abs_err / n_rows
        lo, hi = _bootstrap_ci(cos.tolist())
        return {
            "metric": "action_parity", "value": float(cos.min()), "status": "measured",
            "min_action_cosine": float(cos.min()), "mean_action_cosine": float(cos.mean()),
            "cosine_ci95": [lo, hi],
            "max_per_dim_mae": float(per_dim.max()), "mean_per_dim_mae": float(per_dim.mean()),
            "per_dim_mae": [float(x) for x in per_dim],
            "n_obs": int(len(images)), "n_action_rows": int(n_rows),
            "action_dim": int(refs.shape[2]), "chunk_size": int(refs.shape[1]),
            "sampler": "vqbet_greedy", "deterministic": True,
            "reference": "torch vqbet rollout, multinomial->argmax (greedy), fp32 as loaded",
            "runner": f"coreai-fabric-parity-runner/{__version__}", "environment": _environment(),
        }

    result = asyncio.run(_run())
    emit = out / "action-parity-measured.json"
    emit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nok: wrote {emit} (feeds `coreai-fabric verify {out.name}`)")


def main():
    ap = argparse.ArgumentParser(description="VQ-BeT greedy action_parity")
    ap.add_argument("phase", nargs="?", default="reference", choices=["reference"])
    ap.add_argument("--repo", default="lerobot/vqbet_pusht")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.compare:
        cmd_compare(args.out, args.bundle or (args.out / f"{args.out.name}.aimodel"))
    else:
        cmd_reference(args.repo, args.out, args.n_frames, args.seed)


if __name__ == "__main__":
    main()
