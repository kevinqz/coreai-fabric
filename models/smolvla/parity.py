"""SmolVLA flow-matching action_parity — adapted from models/pi05/parity.py.

TWO-VENV: `reference` (venv-A, .venv-lerobot) writes ref chunks + noise to an .npz; `--compare`
(venv-B, .venv + coreai runtime) drives the SAME observations through the asset's encode +
denoise_step over the identical Euler loop and reports min chunk-cosine + global-RMS normalized
MAE. SmolVLA's encode takes `state` as a separate input (like pi0); denoise_step does not.

  .venv-lerobot/bin/python models/smolvla/parity.py reference --out build/smolvla-so101
  .venv/bin/python         models/smolvla/parity.py --compare  --out build/smolvla-so101 \
        --bundle build/smolvla-so101/smolvla-so101.aimodel
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export as smolvla_export  # noqa: E402 — the wrappers + constants + no_deepcopy

REF_NPZ = "smolvla_parity_ref.npz"


def _bootstrap_ci(vals, n=1000, seed=0):
    import numpy as np
    a = np.asarray(vals, float)
    if len(a) < 2:
        return float(a.min()) if len(a) else 0.0, float(a.max()) if len(a) else 0.0
    g = np.random.default_rng(seed)
    mins = [g.choice(a, len(a), replace=True).min() for _ in range(n)]
    return float(np.percentile(mins, 2.5)), float(np.percentile(mins, 97.5))


def _obs(seed: int, fp):
    """Fixed-seed REALISTIC observation (3 distinct high-entropy 1/f images + state + lang) + fixed
    noise. NOT white noise: torch.rand images are OOD for the vision tower and ill-condition the
    action cosine (min drops 0.99->0.96 as an artifact), which is exactly the regime where the
    export's real-image fidelity is misrepresented. natural_image() matches deployment statistics."""
    import torch
    g = torch.Generator().manual_seed(seed)
    C = smolvla_export
    img = [C.natural_image(g).to(fp) for _ in range(3)]
    imask = [torch.ones(1, dtype=torch.bool) for _ in range(3)]
    tok = torch.zeros(1, C.TOK, dtype=torch.long)
    lmask = torch.ones(1, C.TOK, dtype=torch.bool)
    state = (torch.randn(1, C.STATE_DIM, generator=g) * 0.1).to(fp)
    noise = torch.randn(1, C.CHUNK, C.ACT_DIM, generator=g).to(fp)
    return img, imask, tok, lmask, state, noise


def _euler_torch(enc, den, obs, num_steps):
    import torch
    img, imask, tok, lmask, state, noise = obs
    with torch.no_grad(), smolvla_export.no_deepcopy():
        enc_out = enc(img[0], img[1], img[2], imask[0], imask[1], imask[2], tok, lmask, state)
        # enc_out = (prefix_pad_masks, prefix_embeds anchor, *2*N_LAYERS cache tensors). Skip the
        # prefix_embeds anchor (index 1, only there to fix the coreai lowering) — take the trailing
        # cache tensors so the host-side Euler loop matches the .aimodel's declared cache I/O.
        ppad, cache = enc_out[0], enc_out[-2 * smolvla_export.N_LAYERS:]
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = torch.tensor(1.0 + step * dt, dtype=x_t.dtype).expand(1)
            v_t = den(ppad, x_t, t, *cache)
            x_t = x_t + dt * v_t
    return x_t.float().numpy()


def cmd_reference(out: Path, n_frames: int, seed: int, num_steps: int, fp16: bool = True):
    import numpy as np
    import torch  # noqa: F401
    enc, den, _ = smolvla_export._build_wrappers(fp16=fp16)
    import torch as _t
    fd = _t.float16 if fp16 else _t.float32
    refs, imgs, states, noises = [], [], [], []
    for i in range(n_frames):
        obs = _obs(seed + i, fd)
        ref = _euler_torch(enc, den, obs, num_steps)
        img, imask, tok, lmask, state, noise = obs
        refs.append(ref)
        imgs.append(np.stack([x.float().numpy()[0] for x in img]))   # [3,3,IMG,IMG]
        states.append(state.float().numpy())
        noises.append(noise.float().numpy())
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / REF_NPZ,
             refs=np.stack(refs).astype(np.float32),
             images=np.stack(imgs).astype(np.float32),
             states=np.concatenate(states).astype(np.float32),
             noises=np.concatenate(noises).astype(np.float32),
             num_steps=num_steps, prefix_len=smolvla_export.PREFIX_LEN)
    print(f"ok: wrote {out/REF_NPZ}  (n={n_frames}, {num_steps}-step, chunk={refs[0].shape[1]}x{refs[0].shape[2]})")
    print("next: export --free-weights, --lower, then this script --compare")


def cmd_compare(out: Path, bundle: Path):
    import asyncio
    import numpy as np
    from coreai.runtime import AIModel, NDArray
    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _environment

    d = np.load(out / REF_NPZ, allow_pickle=True)
    refs, images, states, noises = d["refs"], d["images"], d["states"], d["noises"]
    num_steps = int(d["num_steps"])
    C = smolvla_export

    async def _run() -> dict:
        model = await AIModel.load(str(bundle))
        try:
            enc = model.load_function("encode")
            den = model.load_function("denoise_step")
        except Exception:  # noqa: BLE001
            return {"metric": "action_parity", "value": None, "status": "not_run",
                    "reason": "asset lacks encode + denoise_step entrypoints"}

        def nd(fn, name, arr):
            dt = np.dtype(str(fn.desc.input_descriptor(name).dtype))
            return NDArray(np.asarray(arr).astype(dt))

        async def drive(i):
            im = images[i]                                   # [3,3,IMG,IMG]
            enc_in = {"img0": nd(enc, "img0", im[0:1]), "img1": nd(enc, "img1", im[1:2]),
                      "img2": nd(enc, "img2", im[2:3]),
                      "imask0": nd(enc, "imask0", np.ones((1,), bool)),
                      "imask1": nd(enc, "imask1", np.ones((1,), bool)),
                      "imask2": nd(enc, "imask2", np.ones((1,), bool)),
                      "lang_tokens": nd(enc, "lang_tokens", np.zeros((1, C.TOK))),
                      "lang_masks": nd(enc, "lang_masks", np.ones((1, C.TOK))),
                      "state": nd(enc, "state", states[i:i + 1])}
            eo = await enc(inputs=enc_in)
            ppad = eo["prefix_pad_masks"]
            cache = {n: eo[n] for n in C._CACHE_NAMES}
            x_t = noises[i:i + 1].astype(np.float64)
            dt = -1.0 / num_steps
            for step in range(num_steps):
                din = {"prefix_pad_masks": ppad,
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
            per_dim_maes.append(np.abs(a - r).reshape(r.shape[-1], -1).mean(axis=1))
        cos = np.asarray(cosines)
        per_dim = np.mean(per_dim_maes, axis=0)
        lo, hi = _bootstrap_ci(cos.tolist())
        flat = refs.reshape(-1, refs.shape[-1]).astype(np.float64)
        global_rms = float(np.sqrt(np.mean(flat ** 2)))
        max_pd = float(np.max(per_dim_maes))
        return {
            "metric": "action_parity", "value": float(cos.min()), "status": "measured",
            "min_action_cosine": float(cos.min()), "mean_action_cosine": float(cos.mean()),
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
                         "Euler loop, fixed noise (isolates export+optimization fidelity)",
            "runner": f"coreai-fabric-parity-runner/{__version__}", "environment": _environment(),
        }

    result = asyncio.run(_run())
    import json
    emit = out / "action-parity-measured.json"
    emit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nok: wrote {emit} (feeds `coreai-fabric verify {out.name}`)")


def main():
    ap = argparse.ArgumentParser(description="SmolVLA flow-matching action_parity")
    ap.add_argument("phase", nargs="?", default="reference", choices=["reference"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()
    if args.compare:
        cmd_compare(args.out, args.bundle or (args.out / f"{args.out.name}.aimodel"))
    else:
        cmd_reference(args.out, args.n_frames, args.seed, args.num_steps, fp16=not args.fp32)


if __name__ == "__main__":
    main()
