# /// script
# requires-python = ">=3.12"
# ///
"""VLA-JEPA action-head action_parity.

This harness gates the export lane in models/vla_jepa/export.py:

  action_denoise_step(conditioning_tokens, x_t, timestep[, state]) -> velocity

It compares the upstream Torch action head and the lowered Core AI function over
the same fixed conditioning tokens, state, initial noise, and VLA-JEPA Euler
sampler. The JEPA world model is training-only, and Qwen context export is a
separate lane; this proves numeric fidelity of the action denoising graph.

Usage:
  .venv-lerobot/bin/python models/vla_jepa/parity.py reference \
    --config-json build/_vla_jepa/VLA-JEPA-LIBERO/config.json \
    --weights build/_vla_jepa/VLA-JEPA-LIBERO/model.safetensors \
    --out build/vla-jepa-libero
  .venv/bin/python models/vla_jepa/parity.py --compare --out build/vla-jepa-libero
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export as vla_jepa_export  # noqa: E402

REF_NPZ = "vla_jepa_action_head_parity_ref.npz"


def _bootstrap_ci(vals, n=1000, seed=0):
    import numpy as np

    a = np.asarray(vals, float)
    if len(a) < 2:
        return float(a.min()) if len(a) else 0.0, float(a.max()) if len(a) else 0.0
    g = np.random.default_rng(seed)
    mins = [g.choice(a, len(a), replace=True).min() for _ in range(n)]
    return float(np.percentile(mins, 2.5)), float(np.percentile(mins, 97.5))


def _build_reference(config_json: Path, weights: Path | None, has_state_arg: bool | None):
    from lerobot.policies.vla_jepa.action_head import VLAJEPAActionHead

    cfg, raw = vla_jepa_export._load_action_config(config_json, probe_small=False)
    has_state = has_state_arg if has_state_arg is not None else vla_jepa_export._has_state(raw)
    head = VLAJEPAActionHead(
        cfg, cross_attention_dim=vla_jepa_export.QWEN3_VL_2B_HIDDEN
    ).to("cpu").eval()
    vla_jepa_export._load_action_weights(head, weights)
    wrapper = vla_jepa_export.DenoiseStepWrapper.build(head, has_state)
    return cfg, has_state, wrapper.eval()


def _obs(seed: int, cfg, has_state: bool):
    import torch

    g = torch.Generator().manual_seed(seed)
    conditioning = torch.randn(
        1,
        int(cfg.num_embodied_action_tokens_per_instruction),
        vla_jepa_export.QWEN3_VL_2B_HIDDEN,
        generator=g,
        dtype=torch.float32,
    )
    noise = torch.randn(1, int(cfg.chunk_size), int(cfg.action_dim), generator=g, dtype=torch.float32)
    state = None
    if has_state:
        state = (torch.randn(1, 1, int(cfg.state_dim), generator=g) * 0.1).to(torch.float32)
    return conditioning, state, noise


def _euler_torch(wrapper, cfg, obs):
    import torch

    conditioning, state, noise = obs
    num_steps = int(cfg.num_inference_timesteps)
    dt = 1.0 / max(num_steps, 1)
    x_t = noise
    with torch.no_grad():
        for step in range(num_steps):
            t_value = int((step / float(max(num_steps, 1))) * int(cfg.action_num_timestep_buckets))
            timestep = torch.full((1,), t_value, dtype=torch.long)
            if state is None:
                velocity = wrapper(conditioning, x_t, timestep)
            else:
                velocity = wrapper(conditioning, x_t, timestep, state)
            x_t = x_t + dt * velocity
    return x_t.float().numpy()


def cmd_reference(
    out: Path,
    config_json: Path,
    weights: Path | None,
    n_obs: int,
    seed: int,
    has_state_arg: bool | None,
) -> None:
    import numpy as np

    cfg, has_state, wrapper = _build_reference(config_json, weights, has_state_arg)
    refs, conditionings, states, noises = [], [], [], []
    for i in range(n_obs):
        obs = _obs(seed + i, cfg, has_state)
        ref = _euler_torch(wrapper, cfg, obs)
        conditioning, state, noise = obs
        refs.append(ref)
        conditionings.append(conditioning.numpy())
        states.append(state.numpy() if state is not None else np.zeros((1, 0, 0), dtype=np.float32))
        noises.append(noise.numpy())

    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / REF_NPZ,
        refs=np.stack(refs).astype(np.float32),
        conditionings=np.stack(conditionings).astype(np.float32),
        states=np.stack(states).astype(np.float32),
        noises=np.stack(noises).astype(np.float32),
        has_state=bool(has_state),
        num_steps=int(cfg.num_inference_timesteps),
        action_num_timestep_buckets=int(cfg.action_num_timestep_buckets),
        chunk_size=int(cfg.chunk_size),
        action_dim=int(cfg.action_dim),
        state_dim=int(cfg.state_dim),
        seed=int(seed),
    )
    print(f"ok: wrote {out / REF_NPZ} (n={n_obs}, steps={int(cfg.num_inference_timesteps)})")
    print("next: lower the bundle, then run this script with --compare")


def cmd_compare(out: Path, bundle: Path) -> None:
    import asyncio

    import numpy as np
    from coreai.runtime import AIModel, NDArray
    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _environment

    d = np.load(out / REF_NPZ, allow_pickle=True)
    refs = d["refs"]
    conditionings = d["conditionings"]
    states = d["states"]
    noises = d["noises"]
    has_state = bool(d["has_state"])
    num_steps = int(d["num_steps"])
    timestep_buckets = int(d["action_num_timestep_buckets"])

    async def _run() -> dict:
        model = await AIModel.load(str(bundle))
        try:
            den = model.load_function("action_denoise_step")
        except Exception:  # noqa: BLE001
            return {
                "metric": "action_parity",
                "value": None,
                "status": "not_run",
                "reason": "asset lacks action_denoise_step entrypoint",
            }

        def nd(name, arr):
            dt = np.dtype(str(den.desc.input_descriptor(name).dtype))
            return NDArray(np.asarray(arr).astype(dt))

        async def drive(i):
            x_t = noises[i].astype(np.float64)
            dt = 1.0 / max(num_steps, 1)
            for step in range(num_steps):
                t_value = int((step / float(max(num_steps, 1))) * timestep_buckets)
                inputs = {
                    "conditioning_tokens": nd("conditioning_tokens", conditionings[i]),
                    "x_t": nd("x_t", x_t),
                    "timestep": nd("timestep", [t_value]),
                }
                if has_state:
                    inputs["state"] = nd("state", states[i])
                outv = await den(inputs=inputs)
                x_t = x_t + dt * outv["velocity"].numpy().astype(np.float64)
            return x_t

        cosines, per_dim_maes = [], []
        for i in range(len(refs)):
            a = np.asarray(await drive(i), dtype=np.float64)
            r = refs[i].astype(np.float64)
            cosines.append(float(np.dot(r.reshape(-1), a.reshape(-1)) /
                                 (np.linalg.norm(r) * np.linalg.norm(a) + 1e-12)))
            per_dim_maes.append(np.abs(a - r).reshape(-1, r.shape[-1]).mean(axis=0))

        cos = np.asarray(cosines)
        per_dim = np.mean(per_dim_maes, axis=0)
        lo, hi = _bootstrap_ci(cos.tolist())
        flat = refs.reshape(-1, refs.shape[-1]).astype(np.float64)
        global_rms = float(np.sqrt(np.mean(flat ** 2)))
        max_pd = float(np.max(per_dim_maes))
        return {
            "metric": "action_parity",
            "value": float(cos.min()),
            "status": "measured",
            "parity_kind": "vla_jepa_action_head_fixed_noise",
            "min_action_cosine": float(cos.min()),
            "median_action_cosine": float(np.median(cos)),
            "mean_action_cosine": float(cos.mean()),
            "cosine_ci95": [lo, hi],
            "max_per_dim_mae": max_pd,
            "mean_per_dim_mae": float(np.mean(per_dim_maes)),
            "per_dim_mae": [float(x) for x in per_dim],
            "action_rms": global_rms,
            "max_relative_action_mae": (max_pd / global_rms) if global_rms > 1e-9 else 0.0,
            "per_obs_cosine": [float(x) for x in cosines],
            "per_obs_ref_rms": [
                float(np.sqrt(np.mean(refs[i].astype(np.float64) ** 2))) for i in range(len(refs))
            ],
            "n_obs": int(len(refs)),
            "num_steps": num_steps,
            "action_dim": int(refs.shape[-1]),
            "chunk_size": int(refs.shape[-2]),
            "sampler": "vla_jepa_flow_matching_euler",
            "deterministic": True,
            "reference": (
                "Torch VLA-JEPA action head vs Core AI action_denoise_step, identical "
                "synthetic Qwen-context tokens, fixed noise, and Euler loop; this gates "
                "the action-head export lane, not downstream task success."
            ),
            "runner": f"coreai-fabric-parity-runner/{__version__}",
            "environment": _environment(),
        }

    result = asyncio.run(_run())
    emit = out / "action-parity-measured.json"
    emit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nok: wrote {emit} (feeds `coreai-fabric verify {out.name}`)")


def main() -> None:
    ap = argparse.ArgumentParser(description="VLA-JEPA action-head action_parity")
    ap.add_argument("phase", nargs="?", default="reference", choices=["reference"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--config-json", type=Path)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--n-obs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-state", action=argparse.BooleanOptionalAction, default=None)
    args = ap.parse_args()
    if args.compare:
        cmd_compare(args.out, args.bundle or (args.out / f"{args.out.name}.aimodel"))
    else:
        if args.config_json is None:
            raise SystemExit("--config-json is required in reference mode")
        cmd_reference(args.out, args.config_json, args.weights, args.n_obs, args.seed, args.with_state)


if __name__ == "__main__":
    main()
