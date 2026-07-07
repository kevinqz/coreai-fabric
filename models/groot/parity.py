# /// script
# requires-python = ">=3.12"
# ///
"""GR00T-N1.7 action-head action_parity (Gate B).

Drives the flow-matching Euler loop (x_{i+1} = x_i + dt * velocity, dt =
1/num_inference_timesteps) with FIXED synthetic VL context + FIXED initial noise
through BOTH the torch reference action head and the lowered .aimodel, then
compares the final action chunk. Single env (.venv): the head is pure-torch +
diffusers and the coreai runtime runs the asset, so one process produces both
sides. Mirrors the EVO1 lane.

Writes action-parity-measured.json next to the bundle; `coreai-fabric verify`
records it and recomputes pass/fail vs the recipe threshold.

Usage:
  .venv/bin/python models/groot/parity.py --compare \
      --weights build/_groot --out build/groot-n1-7-3b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export as groot_export  # noqa: E402


def _bootstrap_ci(vals, n=1000, seed=0):
    a = np.asarray(vals, float)
    if len(a) < 2:
        return float(a.min()) if len(a) else 0.0, float(a.max()) if len(a) else 0.0
    g = np.random.default_rng(seed)
    mins = [g.choice(a, len(a), replace=True).min() for _ in range(n)]
    return float(np.percentile(mins, 2.5)), float(np.percentile(mins, 97.5))


def _obs(seed):
    import torch
    E = groot_export
    g = torch.Generator().manual_seed(seed)
    vl = torch.randn(1, E.CTX_TOKENS, E.XDIM, generator=g, dtype=torch.float32)
    state = torch.randn(1, 1, E.INNER, generator=g, dtype=torch.float32) * 0.5
    noise = (torch.rand(1, E.HORIZON, E.ACTION_DIM, generator=g, dtype=torch.float32) * 2 - 1)
    return vl, state, noise


def cmd_compare(weights, out, bundle, n_obs, num_steps, seed) -> None:
    import asyncio
    import torch
    from coreai.runtime import AIModel, NDArray
    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _environment

    E = groot_export
    head = E._build_head()
    E._load_action_head_weights(head, weights)
    step = E.DenoiseStep(head).eval()
    emb_id = torch.zeros(1, dtype=torch.long)
    im = torch.ones(1, E.CTX_TOKENS, dtype=torch.bool)
    bm = torch.ones(1, E.CTX_TOKENS, dtype=torch.bool)
    dt = 1.0 / num_steps

    def time_index(i):
        return torch.tensor([min(int((i / num_steps) * E.NUM_TIMESTEP_BUCKETS),
                                 E.NUM_TIMESTEP_BUCKETS - 1)], dtype=torch.long)

    obs = [_obs(seed + i) for i in range(n_obs)]

    # Torch reference: Euler with the torch DenoiseStep.
    refs = []
    with torch.no_grad():
        for vl, state, noise in obs:
            x = noise.clone()
            for i in range(num_steps):
                v = step(x, time_index(i), vl, state, emb_id, im, bm).view(1, E.HORIZON, E.ACTION_DIM)
                x = x + dt * v
            refs.append(x.float().numpy())

    async def _run() -> dict:
        model = await AIModel.load(str(bundle))
        try:
            den = model.load_function("groot_denoise_step")
        except Exception:  # noqa: BLE001
            return {"metric": "action_parity", "value": None, "status": "not_run",
                    "reason": "asset lacks groot_denoise_step entrypoint"}

        def nd(name, arr):
            dt_ = np.dtype(str(den.desc.input_descriptor(name).dtype))
            return NDArray(np.asarray(arr).astype(dt_))

        cosines, per_dim_maes = [], []
        for (vl, state, noise), ref in zip(obs, refs):
            x = noise.numpy().astype(np.float64)
            vlnp, snp = vl.numpy(), state.numpy()
            for i in range(num_steps):
                ti = min(int((i / num_steps) * E.NUM_TIMESTEP_BUCKETS), E.NUM_TIMESTEP_BUCKETS - 1)
                outv = await den(inputs={
                    "x_t": nd("x_t", x),
                    "timestep": nd("timestep", [ti]),
                    "vl_embeds": nd("vl_embeds", vlnp),
                    "state_features": nd("state_features", snp),
                    "embodiment_id": nd("embodiment_id", [0]),
                    "image_mask": nd("image_mask", np.ones((1, E.CTX_TOKENS), bool)),
                    "backbone_attention_mask": nd("backbone_attention_mask", np.ones((1, E.CTX_TOKENS), bool)),
                })
                v = outv["velocity"].numpy().astype(np.float64).reshape(1, x.shape[1], x.shape[2])
                x = x + dt * v
            a = x.reshape(-1)
            r = ref.astype(np.float64).reshape(-1)
            cosines.append(float(np.dot(a, r) / (np.linalg.norm(a) * np.linalg.norm(r) + 1e-12)))
            per_dim_maes.append(np.abs(x - ref.astype(np.float64)).reshape(-1, x.shape[-1]).mean(axis=0))

        cos = np.asarray(cosines)
        per_dim = np.mean(per_dim_maes, axis=0)
        lo, hi = _bootstrap_ci(cos.tolist())
        rms = float(np.sqrt(np.mean(np.stack(refs).astype(np.float64) ** 2)))
        max_pd = float(np.max(per_dim_maes))
        return {
            "metric": "action_parity", "value": float(cos.min()), "status": "measured",
            "parity_kind": "groot_flow_matching_fixed_noise",
            "min_action_cosine": float(cos.min()), "median_action_cosine": float(np.median(cos)),
            "mean_action_cosine": float(cos.mean()), "cosine_ci95": [lo, hi],
            "max_per_dim_mae": max_pd, "mean_per_dim_mae": float(np.mean(per_dim_maes)),
            "per_dim_mae": [float(x) for x in per_dim], "action_rms": rms,
            "max_relative_action_mae": (max_pd / rms) if rms > 1e-9 else 0.0,
            "per_obs_cosine": [float(x) for x in cosines],
            "n_obs": int(len(refs)), "num_steps": num_steps, "action_dim": int(E.ACTION_DIM),
            "chunk_size": int(E.HORIZON), "sampler": "groot_flow_matching_euler", "deterministic": True,
            "reference": ("Torch GR00T-N1.7 flow-matching action head vs Core AI groot_denoise_step, "
                          "identical synthetic VL context + fixed noise + Euler loop; gates the "
                          "action-head export lane, not downstream task success."),
            "runner": f"coreai-fabric-parity-runner/{__version__}", "environment": _environment(),
        }

    result = asyncio.run(_run())
    emit = out / "action-parity-measured.json"
    emit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("value", "min_action_cosine", "median_action_cosine", "max_relative_action_mae", "n_obs", "num_steps") if k in result}, indent=2))
    print(f"ok: wrote {emit}")


def main() -> None:
    ap = argparse.ArgumentParser(description="GR00T action_parity")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--n-obs", type=int, default=8)
    ap.add_argument("--num-steps", type=int, default=groot_export.NUM_STEPS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cmd_compare(args.weights, args.out, args.bundle or (args.out / f"{args.out.name}.aimodel"),
                args.n_obs, args.num_steps, args.seed)


if __name__ == "__main__":
    main()
