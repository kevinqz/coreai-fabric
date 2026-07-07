# /// script
# requires-python = ">=3.12"
# ///
"""EuroBERT graph_output_cosine parity (Gate B for the non-AR encoder lane).

Feeds N seeded `(input_ids, attention_mask)` through BOTH the torch reference and
the lowered `.aimodel`, then compares the per-token logits by cosine. Reports the
MINIMUM cosine across inputs as `value` (one worst input can't hide behind a mean),
plus mean/median + a bootstrap CI. Single env: transformers and coreai_torch
coexist in fabric's .venv, so one process produces both sides.

Writes `graph-output-parity-measured.json` next to the bundle; `coreai-fabric
verify` records it and recomputes pass/fail vs the recipe threshold.

Usage:
  .venv/bin/python models/eurobert/parity.py --compare \
      --model build/_eurobert/pulpie-orange-base --out build/pulpie-orange-base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export as eb_export  # noqa: E402


def _bootstrap_ci(vals, n=1000, seed=0):
    a = np.asarray(vals, float)
    if len(a) < 2:
        return float(a.min()) if len(a) else 0.0, float(a.max()) if len(a) else 0.0
    g = np.random.default_rng(seed)
    mins = [g.choice(a, len(a), replace=True).min() for _ in range(n)]
    return float(np.percentile(mins, 2.5)), float(np.percentile(mins, 97.5))


def cmd_compare(model_ref: str, out: Path, bundle: Path, n_obs: int, seq: int, seed: int) -> None:
    import asyncio

    import torch
    from coreai.runtime import AIModel, NDArray
    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _environment

    cfg, model = eb_export._load(model_ref)
    wrapper = eb_export.LogitsWrapper(model).eval()
    vocab = int(cfg.vocab_size)

    g = torch.Generator().manual_seed(seed)
    ids_list, mask_list, refs = [], [], []
    for _ in range(n_obs):
        ids = torch.randint(0, vocab, (1, seq), generator=g, dtype=torch.long)
        mask = torch.ones(1, seq, dtype=torch.long)
        with torch.no_grad():
            r = wrapper(ids, mask).float().numpy()
        ids_list.append(ids.numpy())
        mask_list.append(mask.numpy())
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
        for ids, mask, r in zip(ids_list, mask_list, refs):
            outv = await fn(inputs={
                "input_ids": nd("input_ids", ids),
                "attention_mask": nd("attention_mask", mask),
            })
            a = outv["logits"].numpy().astype(np.float64).reshape(-1)
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
            "seq_len": int(seq),
            "num_labels": int(refs[0].shape[-1]),
            "reference_dtype": "float32",
            "runner": f"coreai-fabric-parity-runner/{__version__}",
            "environment": _environment(),
            "reference": (
                "Torch EuroBERT token-classification logits vs the Core AI .aimodel over "
                "identical seeded (input_ids, attention_mask); non-autoregressive single forward."
            ),
        }

    result = asyncio.run(_run())
    emit = out / "graph-output-parity-measured.json"
    emit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nok: wrote {emit} (feeds `coreai-fabric verify {out.name}`)")


def main() -> None:
    ap = argparse.ArgumentParser(description="EuroBERT graph_output_cosine parity")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--model", required=True, help="HF id or local snapshot dir (torch reference)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--n-obs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=eb_export.DEFAULT_SEQ)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cmd_compare(
        args.model, args.out,
        args.bundle or (args.out / f"{args.out.name}.aimodel"),
        args.n_obs, args.seq, args.seed,
    )


if __name__ == "__main__":
    main()
