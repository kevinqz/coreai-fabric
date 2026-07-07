# /// script
# requires-python = ">=3.12"
# ///
"""Robometer reward-head graph_output_cosine parity (Gate B, non-AR encoder lane).

Feeds N seeded frame-embedding tensors `[1, T, hidden]` through BOTH the torch
reference reward heads and the lowered `.aimodel`, then compares BOTH outputs
(progress_logits + success_logits, concatenated) by cosine. Reports the MINIMUM
cosine across inputs as `value`. Single env: torch + coreai_torch coexist in
fabric's .venv, so one process produces both sides.

Writes `graph-output-parity-measured.json` next to the bundle; `coreai-fabric
verify` records it and recomputes pass/fail vs the recipe threshold.

Usage:
  .venv/bin/python models/robometer/parity.py --compare \
      --weights build/_robometer/model.safetensors --out build/robometer-4b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export as rbm_export  # noqa: E402


def _bootstrap_ci(vals, n=1000, seed=0):
    a = np.asarray(vals, float)
    if len(a) < 2:
        return float(a.min()) if len(a) else 0.0, float(a.max()) if len(a) else 0.0
    g = np.random.default_rng(seed)
    mins = [g.choice(a, len(a), replace=True).min() for _ in range(n)]
    return float(np.percentile(mins, 2.5)), float(np.percentile(mins, 97.5))


def cmd_compare(weights: Path, out: Path, bundle: Path, n_obs: int, seed: int) -> None:
    import asyncio

    import torch
    from coreai.runtime import AIModel, NDArray
    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _environment

    heads, hidden_dim, bins = rbm_export.build(weights)
    T = rbm_export.MAX_FRAMES

    g = torch.Generator().manual_seed(seed)
    embs, refs = [], []
    for _ in range(n_obs):
        # Seeded per-frame prog-token hidden states (the host reads the real ones out
        # of Qwen3-VL; a fidelity check only needs identical inputs on both sides).
        emb = torch.randn(1, T, hidden_dim, generator=g, dtype=torch.float32)
        with torch.no_grad():
            p, s = heads(emb)
        embs.append(emb.numpy())
        refs.append((p.float().numpy(), s.float().numpy()))

    async def _run() -> dict:
        m = await AIModel.load(str(bundle))
        try:
            fn = m.load_function("reward_heads")
        except Exception:  # noqa: BLE001
            return {"metric": "graph_output_cosine", "value": None, "status": "not_run",
                    "reason": "asset lacks the 'reward_heads' entrypoint"}

        def nd(name, arr):
            dt = np.dtype(str(fn.desc.input_descriptor(name).dtype))
            return NDArray(np.asarray(arr).astype(dt))

        cosines, prog_cos, succ_cos = [], [], []
        for emb, (pr, sr) in zip(embs, refs):
            outv = await fn(inputs={"frame_embeddings": nd("frame_embeddings", emb)})
            pa = outv["progress_logits"].numpy().astype(np.float64).reshape(-1)
            sa = outv["success_logits"].numpy().astype(np.float64).reshape(-1)
            prb = pr.astype(np.float64).reshape(-1)
            srb = sr.astype(np.float64).reshape(-1)

            def _cos(x, y):
                return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))

            pc, sc = _cos(pa, prb), _cos(sa, srb)
            prog_cos.append(pc)
            succ_cos.append(sc)
            # worst of the two heads drives the per-obs cosine
            cosines.append(min(pc, sc))

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
            "min_progress_cosine": float(np.min(prog_cos)),
            "min_success_cosine": float(np.min(succ_cos)),
            "per_obs_cosine": [float(x) for x in cosines],
            "n_obs": int(len(refs)),
            "hidden_dim": int(hidden_dim),
            "progress_bins": int(bins),
            "frames": int(T),
            "reference_dtype": "float32",
            "runner": f"coreai-fabric-parity-runner/{__version__}",
            "environment": _environment(),
            "reference": (
                "Torch Robometer reward heads (progress + success) vs the Core AI .aimodel over "
                "identical seeded per-frame prog-token hidden states; non-autoregressive single "
                "forward (the Qwen3-VL-4B backbone is host-owned, not in the asset). The metric is "
                "the worst per-obs cosine across both output heads."
            ),
        }

    result = asyncio.run(_run())
    emit = out / "graph-output-parity-measured.json"
    emit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nok: wrote {emit} (feeds `coreai-fabric verify {out.name}`)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Robometer graph_output_cosine parity")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--n-obs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cmd_compare(args.weights, args.out, args.bundle or (args.out / f"{args.out.name}.aimodel"),
                args.n_obs, args.seed)


if __name__ == "__main__":
    main()
