# /// script
# requires-python = ">=3.12"
# ///
"""action_parity Gate B for the pi0 split-export .aimodel (robot-free, flow-matching).

pi0 is a VLM + flow-matching action expert. The split export is `encode` (VLM prefix ->
KV cache, run once) + `denoise_step` (the expert, host drives it num_steps times in a
flow-matching Euler loop). Parity drives BOTH sides through the IDENTICAL Euler loop with
FIXED initial noise:
  reference (venv-A): torch encode + torch denoise_step (the export wrappers), fp32
  asset     (venv-B): the .aimodel's encode + denoise_step entrypoints, fp16
so the only difference is the exported+lowered graphs (incl. the fp16 conversion). Compared
in NORMALIZED action space (min chunk-cosine + per-dim MAE + bootstrap CI). Certifies the
export is numerically faithful to the source policy — NOT task success (the card says so).

The Euler loop (from pi0 sample_actions): dt=-1/num_steps; x_t=noise;
for step: time=1+step*dt; v_t=denoise_step(state,ppad,cache,x_t,[time]); x_t += dt*v_t.

TWO-VENV: `reference` (venv-A) writes ref chunks + noise; `--compare` (venv-B) drives the
asset. Run reference BEFORE `export --free-weights` (which deletes the safetensors).

Usage:
  .venv-lerobot/bin/python models/pi0/parity.py reference --out build/pi0-base
  .venv/bin/python         models/pi0/parity.py --compare  --out build/pi0-base
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export as pi0_export  # noqa: E402  (the wrappers + constants + no_deepcopy)

REF_NPZ = "pi0_parity_ref.npz"


def _obs(seed: int, fp):
    """Fixed-seed synthetic observation (3 cams + state + lang) + fixed noise. Deterministic
    given the seed; isolates export fidelity (torch vs asset on the SAME input), not task
    success. fp = torch float dtype for the float inputs."""
    import torch
    g = torch.Generator().manual_seed(seed)
    C = pi0_export
    img = [torch.rand(1, 3, 224, 224, generator=g).to(fp) for _ in range(3)]
    imask = [torch.ones(1, dtype=torch.bool) for _ in range(3)]
    tok = torch.zeros(1, C.TOK, dtype=torch.long)
    lmask = torch.ones(1, C.TOK, dtype=torch.bool)
    state = (torch.randn(1, C.STATE_DIM, generator=g) * 0.1).to(fp)
    noise = torch.randn(1, C.CHUNK, C.ACT_DIM, generator=g).to(fp)
    return img, imask, tok, lmask, state, noise


def _euler_torch(enc, den, obs, num_steps):
    """Reference: drive the torch wrappers through the flow-matching Euler loop."""
    import torch
    img, imask, tok, lmask, state, noise = obs
    with torch.no_grad(), pi0_export.no_deepcopy():
        enc_out = enc(img[0], img[1], img[2], imask[0], imask[1], imask[2], tok, lmask)
        ppad, cache = enc_out[0], enc_out[1:]
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = torch.tensor(1.0 + step * dt, dtype=x_t.dtype).expand(1)
            v_t = den(ppad, x_t, t, *cache)      # pi05 denoise drops state
            x_t = x_t + dt * v_t
    return x_t.float().numpy()


def cmd_reference(out: Path, n_frames: int, seed: int, num_steps: int, fp16: bool = True):
    """venv-A: torch encode + denoise Euler loop -> reference action chunks. fp16 (default)
    matches the fp16 asset so parity isolates export/lowering error (not fp16 quantization);
    --fp32 falls back if fp16 CPU torch chokes on an op."""
    import numpy as np
    import torch  # noqa: F401
    enc, den, _ = pi0_export._build_wrappers(fp16=fp16)
    import torch as _t
    fd = _t.float16 if fp16 else _t.float32
    refs, imgs, states, noises = [], [], [], []
    for i in range(n_frames):
        obs = _obs(seed + i, fd)                        # reference compute dtype (matches asset)
        ref = _euler_torch(enc, den, obs, num_steps)
        img, imask, tok, lmask, state, noise = obs
        refs.append(ref)
        imgs.append(np.stack([x.float().numpy()[0] for x in img]))   # [3,3,224,224]
        states.append(state.float().numpy())
        noises.append(noise.float().numpy())
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / REF_NPZ,
             refs=np.stack(refs).astype(np.float32),        # [N,1,chunk,act]
             images=np.stack(imgs).astype(np.float32),      # [N,3,3,224,224]
             states=np.concatenate(states).astype(np.float32),
             noises=np.concatenate(noises).astype(np.float32),
             num_steps=num_steps, prefix_len=pi0_export.PREFIX_LEN)
    print(f"ok: wrote {out/REF_NPZ}  (n={n_frames}, {num_steps}-step flow-matching, "
          f"chunk={refs[0].shape[1]}x{refs[0].shape[2]})")
    print("next: export --free-weights, --lower, then this script --compare")


def cmd_compare(out: Path, bundle: Path):
    """venv-B: drive the asset's encode + denoise_step through the same Euler loop, compare."""
    import asyncio

    import numpy as np
    from coreai.runtime import AIModel, NDArray

    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _bootstrap_ci, _environment

    d = np.load(out / REF_NPZ, allow_pickle=True)
    refs, images, states, noises = d["refs"], d["images"], d["states"], d["noises"]
    num_steps = int(d["num_steps"])
    C = pi0_export

    async def _run() -> dict:
        model = await AIModel.load(str(bundle))
        try:
            enc = model.load_function("encode")
            den = model.load_function("denoise_step")
        except Exception:  # noqa: BLE001
            return {"metric": "action_parity", "value": None, "status": "not_run",
                    "reason": "asset lacks encode + denoise_step entrypoints"}
        # Cast each input to the asset's OWN traced dtype (an fp16-exported asset needs fp16
        # x_t/state/imgs/timestep; coreai may also narrow int64 tokens to int32). Never assume.
        def nd(fn, name, arr):
            dt = np.dtype(str(fn.desc.input_descriptor(name).dtype))
            return NDArray(np.asarray(arr).astype(dt))

        async def drive(i):
            im = images[i]                                   # [3,3,224,224]
            enc_in = {"img0": nd(enc, "img0", im[0:1]), "img1": nd(enc, "img1", im[1:2]),
                      "img2": nd(enc, "img2", im[2:3]),
                      "imask0": nd(enc, "imask0", np.ones((1,), bool)),
                      "imask1": nd(enc, "imask1", np.ones((1,), bool)),
                      "imask2": nd(enc, "imask2", np.ones((1,), bool)),
                      "lang_tokens": nd(enc, "lang_tokens", np.zeros((1, C.TOK))),
                      "lang_masks": nd(enc, "lang_masks", np.ones((1, C.TOK)))}
            eo = await enc(inputs=enc_in)
            ppad = eo["prefix_pad_masks"]
            cache = {n: eo[n] for n in C._CACHE_NAMES}     # already the asset's dtype (its outputs)
            x_t = noises[i:i + 1].astype(np.float64)
            dt = -1.0 / num_steps
            for step in range(num_steps):
                din = {"prefix_pad_masks": ppad,           # pi05 denoise drops state
                       "x_t": nd(den, "x_t", x_t), "timestep": nd(den, "timestep", [1.0 + step * dt])}
                din.update({n: cache[n] for n in C._CACHE_NAMES})
                vo = await den(inputs=din)
                v_t = vo["velocity"].numpy().astype(np.float64)
                x_t = x_t + dt * v_t
            return x_t

        cosines, per_dim_maes = [], []
        for i in range(len(images)):
            a = np.asarray(await drive(i), dtype=np.float64)
            r = refs[i].astype(np.float64)
            cosines.append(float(np.dot(r.reshape(-1), a.reshape(-1)) /
                                 (np.linalg.norm(r) * np.linalg.norm(a) + 1e-12)))
            per_dim_maes.append(np.abs(a - r).reshape(-1, r.shape[-1]).mean(axis=0))
        cos = np.asarray(cosines)
        per_dim = np.mean(per_dim_maes, axis=0)
        lo, hi = _bootstrap_ci(cos.tolist())
        # Scale-invariant normalized MAE = worst per-dim MAE / GLOBAL action RMS (see models/pi0/
        # parity.py for the rationale: absolute MAE fails across checkpoints; per-dim-relative
        # explodes on near-zero dims; global-RMS is robust to both). Complements the cosine gate.
        flat = refs.reshape(-1, refs.shape[-1]).astype(np.float64)
        global_rms = float(np.sqrt(np.mean(flat ** 2)))
        max_pd = float(np.max(per_dim_maes))
        return {
            "metric": "action_parity", "value": float(cos.min()), "status": "measured",
            "min_action_cosine": float(cos.min()), "mean_action_cosine": float(cos.mean()),
            "median_action_cosine": float(np.median(cos)),
            "cosine_ci95": [lo, hi],
            "max_per_dim_mae": max_pd, "mean_per_dim_mae": float(np.mean(per_dim_maes)),
            "per_dim_mae": [float(x) for x in per_dim],
            "action_rms": global_rms,
            "max_relative_action_mae": (max_pd / global_rms) if global_rms > 1e-9 else 0.0,
            "per_obs_cosine": [float(x) for x in cosines],
            "per_obs_ref_rms": [float(np.sqrt(np.mean(refs[i].astype(np.float64) ** 2))) for i in range(len(images))],
            "n_obs": int(len(images)), "num_steps": num_steps,
            "action_dim": int(refs.shape[-1]), "chunk_size": int(refs.shape[-2]),
            "sampler": "flow_matching_euler", "deterministic": True,
            "reference": "torch fp16 encode+denoise vs the fp16 coreai-optimized asset, identical "
                         "10-step Euler loop, fixed noise (isolates export+optimization fidelity)",
            "runner": f"coreai-fabric-parity-runner/{__version__}", "environment": _environment(),
        }

    result = asyncio.run(_run())
    emit = out / "action-parity-measured.json"
    emit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nok: wrote {emit} (feeds `coreai-fabric verify {out.name}`)")


def main():
    ap = argparse.ArgumentParser(description="pi0 flow-matching action_parity")
    ap.add_argument("phase", nargs="?", default="reference", choices=["reference"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--fp32", action="store_true", help="fp32 reference (fallback if fp16 CPU chokes)")
    args = ap.parse_args()
    if args.compare:
        cmd_compare(args.out, args.bundle or (args.out / f"{args.out.name}.aimodel"))
    else:
        cmd_reference(args.out, args.n_frames, args.seed, args.num_steps, fp16=not args.fp32)


if __name__ == "__main__":
    main()
