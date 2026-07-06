# /// script
# requires-python = ">=3.12"
# ///
"""action_parity Gate B for a Diffusion Policy split-export .aimodel (robot-free).

The multi-step SAMPLER analog of ACT's parity: a Diffusion Policy is NOT a single
forward pass, so we prove the export by running the SAME deterministic DDPM host
loop on both sides and comparing the produced action trajectory —
  reference (venv-A): torch `encode` + torch `denoise_step`, fixed initial noise
  asset     (venv-B): the .aimodel's encode + denoise_step entrypoints, same noise
The host loop is deterministic (posterior-mean DDPM, per-step variance = 0) so the
only difference between the two runs is the exported graphs. This isolates NUMERICAL
EXPORT FIDELITY — it does NOT certify sampler stochasticity or real-world task
success (the card says so). It also validates the net-new host sampler that pi0 needs.

Compared in NORMALIZED action space (the analog of pre-detok logits) — un-normalization
is a separate host step (upstream processor stats). min chunk-cosine is the headline,
plus per-dim MAE and a bootstrap CI.

TWO-VENV: `reference` (venv-A) writes ref trajectories + noise + sampler consts to an
.npz; `--compare` (venv-B) drives the asset. See docs/vla-export-runbook.md.

Usage:
  .venv-lerobot/bin/python models/diffusion/parity.py reference --repo lerobot/diffusion_pusht --out build/diffusion-pusht
  .venv/bin/python         models/diffusion/parity.py --compare  --out build/diffusion-pusht
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REF_NPZ = "diffusion_parity_ref.npz"
OBS_STATE = "observation.state"
OBS_IMAGES = "observation.images"


def _ddpm_update(x, eps, t, alphas_cumprod, step_ratio, clip, clip_range):
    """ONE deterministic DDPM step (epsilon pred, posterior mean, variance=0).

    Reproduces diffusers' DDPMScheduler.step closed form with the stochastic noise
    term removed. Shared by the reference (torch eps) and asset (.aimodel eps) loops
    so both run bit-identical math — the ONLY difference is where eps comes from."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64)
    a_t = float(alphas_cumprod[t])
    t_prev = t - step_ratio
    a_prev = float(alphas_cumprod[t_prev]) if t_prev >= 0 else 1.0
    beta_prod_t = 1.0 - a_t
    pred_x0 = (x - (beta_prod_t ** 0.5) * eps) / (a_t ** 0.5)
    if clip:
        pred_x0 = np.clip(pred_x0, -clip_range, clip_range)
    cur_alpha_t = a_t / a_prev
    cur_beta_t = 1.0 - cur_alpha_t
    coef_x0 = (a_prev ** 0.5) * cur_beta_t / beta_prod_t
    coef_xt = (cur_alpha_t ** 0.5) * (1.0 - a_prev) / beta_prod_t
    return coef_x0 * pred_x0 + coef_xt * x       # + variance*noise, variance term = 0


def _ddpm_deterministic(eps_fn, noise, timesteps, alphas_cumprod, step_ratio,
                        clip, clip_range):
    """Reference loop: eps_fn(x_t[np], t[int]) -> eps[np]. Returns x_0 [1,horizon,adim]."""
    import numpy as np
    x = np.asarray(noise, dtype=np.float64).copy()
    for t in timesteps:
        eps = eps_fn(x, int(t))
        x = _ddpm_update(x, eps, int(t), alphas_cumprod, step_ratio, clip, clip_range)
    return x


def cmd_reference(repo: str, out: Path, n_obs: int, seed: int):
    """venv-A: torch encode + torch denoise, deterministic loop -> ref trajectories."""
    import numpy as np
    import torch
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    policy = DiffusionPolicy.from_pretrained(repo).to("cpu").eval()
    dm = policy.diffusion
    cfg = dm.config
    n_obs_steps, horizon = int(cfg.n_obs_steps), int(cfg.horizon)
    adim = int(cfg.action_feature.shape[0])
    sdim = int(cfg.input_features[OBS_STATE].shape[0])
    img_key = next(k for k, f in cfg.input_features.items() if f.type == "VISUAL")
    C, H, W = tuple(cfg.input_features[img_key].shape)
    ncam = len(cfg.image_features)

    dm.noise_scheduler.set_timesteps(dm.num_inference_steps)
    timesteps = [int(t) for t in dm.noise_scheduler.timesteps.tolist()]
    alphas_cumprod = np.asarray(dm.noise_scheduler.alphas_cumprod.tolist(), dtype=np.float64)
    step_ratio = int(cfg.num_train_timesteps) // int(dm.num_inference_steps)

    def torch_encode(images_np, state_np):
        with torch.no_grad():
            return dm._prepare_global_conditioning({
                OBS_IMAGES: torch.from_numpy(images_np).float(),
                OBS_STATE: torch.from_numpy(state_np).float(),
            }).numpy()

    def make_eps_fn(global_cond_np):
        gc = torch.from_numpy(global_cond_np).float()
        def eps_fn(x_np, t):
            with torch.no_grad():
                xt = torch.from_numpy(x_np).float()
                ts = torch.full((xt.shape[0],), t, dtype=torch.long)
                return dm.unet(xt, ts, global_cond=gc).numpy()
        return eps_fn

    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(1, horizon, adim, generator=g).numpy()

    images_all, states_all, refs = [], [], []
    for i in range(n_obs):
        gi = torch.Generator().manual_seed(seed + 1 + i)
        images = torch.rand(1, n_obs_steps, ncam, C, H, W, generator=gi).numpy()
        state = (torch.randn(1, n_obs_steps, sdim, generator=gi) * 0.1).numpy()
        gc = torch_encode(images, state)
        traj = _ddpm_deterministic(make_eps_fn(gc), noise, timesteps, alphas_cumprod,
                                   step_ratio, bool(cfg.clip_sample), float(cfg.clip_sample_range))
        images_all.append(images); states_all.append(state); refs.append(traj)

    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / REF_NPZ,
        images=np.concatenate(images_all).astype(np.float32),
        states=np.concatenate(states_all).astype(np.float32),
        noise=noise.astype(np.float32),
        refs=np.stack(refs).astype(np.float32),           # [N,1,horizon,adim]
        timesteps=np.asarray(timesteps, dtype=np.int64),
        alphas_cumprod=alphas_cumprod,
        step_ratio=step_ratio,
        clip=bool(cfg.clip_sample),
        clip_range=float(cfg.clip_sample_range),
        num_steps=int(dm.num_inference_steps),
    )
    print(f"ok: wrote {out/REF_NPZ}  (n_obs={n_obs}, {dm.num_inference_steps}-step DDPM, "
          f"traj {horizon}x{adim})")
    print("next (venv-B): .venv/bin/python models/diffusion/parity.py --compare --out", out)


def cmd_compare(out: Path, bundle: Path):
    """venv-B: drive the asset's encode + denoise_step in the same loop, compare."""
    import asyncio

    import numpy as np
    from coreai.runtime import AIModel, NDArray

    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _bootstrap_ci, _environment

    d = np.load(out / REF_NPZ, allow_pickle=True)
    images, states, noise, refs = d["images"], d["states"], d["noise"], d["refs"]
    timesteps = [int(t) for t in d["timesteps"].tolist()]
    alphas_cumprod = d["alphas_cumprod"]
    step_ratio, clip, clip_range = int(d["step_ratio"]), bool(d["clip"]), float(d["clip_range"])

    async def _run() -> dict:
        model = await AIModel.load(str(bundle))
        try:
            enc = model.load_function("encode")
            den = model.load_function("denoise_step")
        except Exception:  # noqa: BLE001
            return {"metric": "action_parity", "value": None, "status": "not_run",
                    "reason": "asset does not expose encode + denoise_step entrypoints — "
                              "diffusion action_parity needs the split-export contract."}
        din = list(den.desc.input_names)
        x_name = next((n for n in din if n in ("x_t", "sample")), din[0])
        t_name = next((n for n in din if "time" in n.lower() or n in ("t", "timestep")), None)
        c_name = next((n for n in din if "cond" in n.lower()), None)
        e_out = "eps" if "eps" in den.desc.output_names else den.desc.output_names[0]
        enc_in = list(enc.desc.input_names)
        img_name = next(n for n in enc_in if len(enc.desc.input_descriptor(n).shape) >= 5)
        st_name = next(n for n in enc_in if n != img_name)
        g_out = "global_cond" if "global_cond" in enc.desc.output_names else enc.desc.output_names[0]

        async def asset_encode(images_np, state_np):
            res = await enc(inputs={img_name: NDArray(images_np.astype(np.float32)),
                                    st_name: NDArray(state_np.astype(np.float32))})
            return res[g_out].numpy()

        cosines, per_dim_maes = [], []
        for i in range(len(images)):
            gc = await asset_encode(images[i:i + 1], states[i:i + 1])
            traj = await _drive_ddpm(den, x_name, t_name, c_name, e_out, gc, noise,
                                     timesteps, alphas_cumprod, step_ratio, clip, clip_range)
            r = refs[i].reshape(-1)
            a = np.asarray(traj, dtype=np.float64).reshape(-1)
            cosines.append(float(np.dot(r, a) / (np.linalg.norm(r) * np.linalg.norm(a) + 1e-12)))
            per_dim_maes.append(np.abs(np.asarray(traj) - refs[i]).reshape(-1, refs[i].shape[-1]).mean(axis=0))

        cos = np.asarray(cosines)
        per_dim = np.mean(per_dim_maes, axis=0)
        lo, hi = _bootstrap_ci(cos.tolist())
        return {
            "metric": "action_parity", "value": float(cos.min()), "status": "measured",
            "min_action_cosine": float(cos.min()), "mean_action_cosine": float(cos.mean()),
            "cosine_ci95": [lo, hi],
            "max_per_dim_mae": float(np.max(per_dim_maes)), "mean_per_dim_mae": float(np.mean(per_dim_maes)),
            "per_dim_mae": [float(x) for x in per_dim],
            "n_obs": int(len(images)), "num_steps": int(d["num_steps"]),
            "action_dim": int(refs.shape[-1]), "chunk_size": int(refs.shape[-2]),
            "sampler": "ddpm_deterministic", "deterministic": True,
            "reference": "torch encode + denoise, identical deterministic DDPM host loop",
            "runner": f"coreai-fabric-parity-runner/{__version__}", "environment": _environment(),
        }

    result = asyncio.run(_run())
    emit = out / "action-parity-measured.json"
    emit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nok: wrote {emit} (feeds `coreai-fabric verify {out.name}`)")


async def _drive_ddpm(den, x_name, t_name, c_name, e_out, gc, noise,
                      timesteps, alphas_cumprod, step_ratio, clip, clip_range):
    """Async DDPM loop calling the asset's denoise_step each step (host sampler)."""
    import numpy as np
    from coreai.runtime import NDArray
    x = np.asarray(noise, dtype=np.float64).copy()
    for t in timesteps:
        inp = {x_name: NDArray(x.astype(np.float32)), c_name: NDArray(gc.astype(np.float32))}
        if t_name:
            inp[t_name] = NDArray(np.asarray([t], dtype=np.int32))
        res = await den(inputs=inp)
        eps = res[e_out].numpy()
        x = _ddpm_update(x, eps, int(t), alphas_cumprod, step_ratio, clip, clip_range)
    return x


def main():
    ap = argparse.ArgumentParser(description="Diffusion Policy action_parity")
    ap.add_argument("phase", nargs="?", default="reference", choices=["reference"])
    ap.add_argument("--repo", default="lerobot/diffusion_pusht")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--n-obs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.compare:
        cmd_compare(args.out, args.bundle or (args.out / f"{args.out.name}.aimodel"))
    else:
        cmd_reference(args.repo, args.out, args.n_obs, args.seed)


if __name__ == "__main__":
    main()
