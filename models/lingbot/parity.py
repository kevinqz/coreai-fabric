# /// script
# requires-python = ">=3.12"
# ///
"""LingBot-Vision graph_output_cosine parity (Gate B for the ViT encoder lane).

Feeds N seeded images `[1,3,S,S]` through BOTH the torch reference backbone and
the lowered `.aimodel`, then compares the normalized per-patch tokens by cosine.
Reports the MINIMUM cosine across inputs as `value` (one worst input can't hide
behind a mean), plus mean/median + a bootstrap CI. Single env: torch and
coreai_torch coexist in fabric's .venv, so one process produces both sides.

Writes `graph-output-parity-measured.json` next to the bundle; `coreai-fabric
verify` records it and recomputes pass/fail vs the recipe threshold.

Usage:
  .venv/bin/python models/lingbot/parity.py --compare \
      --variant small --lingbot-src build/_src_lingbot --out build/lingbot-vision-vit-small
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export as lb_export  # noqa: E402


def _bootstrap_ci(vals, n=1000, seed=0):
    a = np.asarray(vals, float)
    if len(a) < 2:
        return float(a.min()) if len(a) else 0.0, float(a.max()) if len(a) else 0.0
    g = np.random.default_rng(seed)
    mins = [g.choice(a, len(a), replace=True).min() for _ in range(n)]
    return float(np.percentile(mins, 2.5)), float(np.percentile(mins, 97.5))


def cmd_compare(variant: str, lingbot_src: Path, out: Path, bundle: Path,
                n_obs: int, size: int, seed: int) -> None:
    import asyncio

    import torch
    from coreai.runtime import AIModel, NDArray
    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _environment

    backbone, embed_dim = lb_export._load_backbone(variant, lingbot_src)
    wrapper = lb_export.PatchTokens(backbone).eval()
    size = int(size) // lb_export.PATCH * lb_export.PATCH

    g = torch.Generator().manual_seed(seed)
    imgs, refs = [], []
    for _ in range(n_obs):
        # Distinct seeded images in a plausible normalized range (host does the real
        # ImageNet normalize; a fidelity check only needs identical inputs both sides).
        img = torch.randn(1, 3, size, size, generator=g, dtype=torch.float32)
        with torch.no_grad():
            r = wrapper(img).float().numpy()
        imgs.append(img.numpy())
        refs.append(r)

    async def _run() -> dict:
        m = await AIModel.load(str(bundle))
        try:
            fn = m.load_function("main")
        except Exception:  # noqa: BLE001
            return {"metric": "graph_output_cosine", "value": None, "status": "not_run",
                    "reason": "asset lacks the 'main' entrypoint"}

        def nd(name, arr):
            dt = np.dtype(str(fn.desc.input_descriptor(name).dtype))
            return NDArray(np.asarray(arr).astype(dt))

        cosines = []
        for img, r in zip(imgs, refs):
            outv = await fn(inputs={"image": nd("image", img)})
            a = outv["patch_tokens"].numpy().astype(np.float64).reshape(-1)
            b = r.astype(np.float64).reshape(-1)
            cosines.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))

        cos = np.asarray(cosines)
        lo, hi = _bootstrap_ci(cos.tolist())
        return {
            "metric": "graph_output_cosine",
            "value": float(cos.min()),
            "status": "measured",
            "min_cosine": float(cos.min()),
            "median_cosine": float(np.median(cos)),
            "mean_cosine": float(cos.mean()),
            "cosine_ci95": [lo, hi],
            "per_obs_cosine": [float(x) for x in cosines],
            "n_obs": int(len(refs)),
            "image_size": int(size),
            "num_patch_tokens": int(refs[0].shape[-2]),
            "embed_dim": int(refs[0].shape[-1]),
            "reference_dtype": "float32",
            "runner": f"coreai-fabric-parity-runner/{__version__}",
            "environment": _environment(),
            "reference": (
                "Torch LingBot-Vision ViT patch tokens vs the Core AI .aimodel over identical "
                "seeded images; non-autoregressive single forward (frozen backbone)."
            ),
        }

    result = asyncio.run(_run())
    emit = out / "graph-output-parity-measured.json"
    emit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nok: wrote {emit} (feeds `coreai-fabric verify {out.name}`)")


def main() -> None:
    ap = argparse.ArgumentParser(description="LingBot-Vision graph_output_cosine parity")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--variant", required=True, choices=["small", "base", "large", "giant"])
    ap.add_argument("--lingbot-src", type=Path, default=Path("build/_src_lingbot"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--n-obs", type=int, default=8)
    ap.add_argument("--size", type=int, default=lb_export.DEFAULT_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cmd_compare(
        args.variant, args.lingbot_src, args.out,
        args.bundle or (args.out / f"{args.out.name}.aimodel"),
        args.n_obs, args.size, args.seed,
    )


if __name__ == "__main__":
    main()
