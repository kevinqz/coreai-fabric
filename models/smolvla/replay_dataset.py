"""SmolVLA REAL-DATA action_parity — the credibility upgrade over synthetic natural_image parity.

Feeds REAL recorded SO-101 robot camera frames + proprioceptive state (from a LeRobot dataset) through BOTH the torch
reference policy graphs and the lowered Core AI asset, over the identical fixed-noise Euler loop, and
compares the predicted action chunk. This proves the export reproduces the source policy on REAL robot
visual/state observations — not just synthetic 1/f images. Language tokens are the documented
zero-token baseline used by models/smolvla/parity.py, not the dataset task string. It is a
conversion-FIDELITY metric (coreai vs torch), NOT task success.

Honest caveat: the SmolVLA-SO101 policy (edge-inference/smolvla-so101-pick-orange) was trained with
`front`+`wrist` cameras; the public `lerobot/svla_so101_pickplace` dataset records `up`+`side`. We map
up->cam0, side->cam1 (+ an empty 3rd SmolVLA slot). So the frames are REAL SO-101 robot pixels from a
DIFFERENT rig than the training set — an OOD-but-real fidelity stress test.

  venv-A: .venv-lerobot/bin/python models/smolvla/replay_dataset.py reference --out build/_smolvla_replay \
            --dataset lerobot/svla_so101_pickplace --episode 0 --n-frames 8
  venv-B: .venv/bin/python         models/smolvla/replay_dataset.py --compare  --out build/_smolvla_replay \
            --bundle build/_smolvla_replay/smolvla-so101.aimodel
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export as C  # noqa: E402 — wrappers + constants (IMG, TOK, STATE_DIM, CHUNK, ACT_DIM, N_LAYERS, ...)

REF_NPZ = "smolvla_replay_ref.npz"
ACTION_PARITY_JSON = "action-parity-measured.json"
REAL_FRAME_PARITY_JSON = "real-frame-parity.json"
# dataset camera key -> SmolVLA slot (the policy's front/wrist; the 3rd SmolVLA slot is empty)
DS_CAMS = ["observation.images.up", "observation.images.side"]


def _prep_image(v):
    """A LeRobot frame image (CHW float[0,1] or HWC uint8) -> [1,3,IMG,IMG] in [-1,1] (the SmolVLA
    range), exactly matching export.py's real-image preprocessing (resize BILINEAR + /127.5 - 1)."""
    import numpy as np
    from PIL import Image
    arr = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:      # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (arr * 255.0).clip(0, 255).astype("uint8") if arr.max() <= 1.01 else arr.clip(0, 255).astype("uint8")
    im = Image.fromarray(arr).convert("RGB").resize((C.IMG, C.IMG), Image.BILINEAR)
    import numpy as _np
    a = _np.asarray(im).astype("float32")
    import torch
    return torch.from_numpy(a).permute(2, 0, 1)[None] / 127.5 - 1.0    # [1,3,IMG,IMG], [-1,1]


def _dataset_obs(batch, seed, fp):
    """REAL-frame obs tuple matching the synthetic _obs contract: (img[3], imask[3], tok, lmask, state,
    noise). cam0=up cam1=side (real, imask=1); cam2=empty (imask=0). state = recorded, padded to 32."""
    import numpy as np
    import torch
    imgs = [_prep_image(batch[k]).to(fp) for k in DS_CAMS]
    imgs.append(torch.zeros(1, 3, C.IMG, C.IMG, dtype=fp))            # empty 3rd SmolVLA camera slot
    imask = [torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)]
    sv = batch["observation.state"]
    st = np.asarray(sv.numpy() if hasattr(sv, "numpy") else sv, dtype="float32").reshape(-1)
    state = torch.zeros(1, C.STATE_DIM, dtype=fp)
    state[0, :min(len(st), C.STATE_DIM)] = torch.from_numpy(st[:C.STATE_DIM]).to(fp)
    tok = torch.zeros(1, C.TOK, dtype=torch.long)                     # zero lang (matches the baseline harness)
    lmask = torch.ones(1, C.TOK, dtype=torch.bool)
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(1, C.CHUNK, C.ACT_DIM, generator=g).to(fp)
    return imgs, imask, tok, lmask, state, noise


def _euler_torch(enc, den, obs, num_steps):
    import torch
    img, imask, tok, lmask, state, noise = obs
    with torch.no_grad(), C.no_deepcopy():
        enc_out = enc(img[0], img[1], img[2], imask[0], imask[1], imask[2], tok, lmask, state)
        ppad, cache = enc_out[0], enc_out[-2 * C.N_LAYERS:]
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            t = torch.tensor(1.0 + step * dt, dtype=x_t.dtype).expand(1)
            v_t = den(ppad, x_t, t, *cache)
            x_t = x_t + dt * v_t
    return x_t.float().numpy()


def cmd_reference(out: Path, dataset: str, episode: int, n_frames: int, seed: int, num_steps: int, fp16=True):
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    enc, den, _ = C._build_wrappers(fp16=fp16)
    fd = torch.float16 if fp16 else torch.float32
    ds = LeRobotDataset(dataset, episodes=[episode])
    # spread frames across the episode (not consecutive near-duplicates)
    idxs = np.linspace(0, len(ds) - 1, num=min(n_frames, len(ds)), dtype=int).tolist()
    refs, imgs, states, noises = [], [], [], []
    for k, i in enumerate(idxs):
        obs = _dataset_obs(ds[int(i)], seed + k, fd)
        ref = _euler_torch(enc, den, obs, num_steps)
        img, imask, tok, lmask, state, noise = obs
        refs.append(ref)
        imgs.append(np.stack([x.float().numpy()[0] for x in img]))
        states.append(state.float().numpy())
        noises.append(noise.float().numpy())
        print(f"  frame {k} (ds idx {i}): ref chunk {ref.shape}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / REF_NPZ, refs=np.stack(refs).astype(np.float32), images=np.stack(imgs).astype(np.float32),
             states=np.concatenate(states).astype(np.float32), noises=np.concatenate(noises).astype(np.float32),
             num_steps=num_steps, dataset=dataset, episode=episode)
    print(f"ok: wrote {out/REF_NPZ} (n={len(idxs)}, REAL frames from {dataset} ep{episode})")


def cmd_compare(out: Path, bundle: Path):
    import asyncio
    import json
    import numpy as np
    from coreai.runtime import AIModel, NDArray
    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _environment
    d = np.load(out / REF_NPZ, allow_pickle=True)
    refs, images, states, noises = d["refs"], d["images"], d["states"], d["noises"]
    num_steps = int(d["num_steps"])

    async def _run() -> dict:
        model = await AIModel.load(str(bundle))
        enc = model.load_function("encode")
        den = model.load_function("denoise_step")

        def nd(fn, name, arr):
            dt = np.dtype(str(fn.desc.input_descriptor(name).dtype))
            return NDArray(np.asarray(arr).astype(dt))

        async def drive(i):
            im = images[i]
            enc_in = {"img0": nd(enc, "img0", im[0:1]), "img1": nd(enc, "img1", im[1:2]), "img2": nd(enc, "img2", im[2:3]),
                      "imask0": nd(enc, "imask0", np.ones((1,), bool)), "imask1": nd(enc, "imask1", np.ones((1,), bool)),
                      "imask2": nd(enc, "imask2", np.zeros((1,), bool)),
                      "lang_tokens": nd(enc, "lang_tokens", np.zeros((1, C.TOK))),
                      "lang_masks": nd(enc, "lang_masks", np.ones((1, C.TOK))),
                      "state": nd(enc, "state", states[i:i + 1])}
            eo = await enc(inputs=enc_in)
            ppad = eo["prefix_pad_masks"]
            cache = {n: eo[n] for n in C._CACHE_NAMES}
            x_t = noises[i:i + 1].astype(np.float64)
            dt = -1.0 / num_steps
            for step in range(num_steps):
                din = {"prefix_pad_masks": ppad, "x_t": nd(den, "x_t", x_t), "timestep": nd(den, "timestep", [1.0 + step * dt])}
                din.update({n: cache[n] for n in C._CACHE_NAMES})
                vo = await den(inputs=din)
                x_t = x_t + dt * vo["velocity"].numpy().astype(np.float64)
            return x_t

        cosines, per_dim_maes, per_obs_ref_rms = [], [], []
        for i in range(len(images)):
            a = np.asarray(await drive(i), np.float64); r = refs[i].astype(np.float64)
            cos = float(np.dot(r.reshape(-1), a.reshape(-1)) / (np.linalg.norm(r) * np.linalg.norm(a) + 1e-12))
            per_dim_maes.append(np.abs(a - r).reshape(-1, r.shape[-1]).mean(axis=0))
            per_obs_ref_rms.append(float(np.sqrt(np.mean(r ** 2))))
            cosines.append(cos); print(f"  frame {i}: real-frame action_cosine {cos:.5f}", flush=True)
        cos = np.asarray(cosines)
        per_dim = np.mean(per_dim_maes, axis=0)
        max_pd = float(np.max(per_dim_maes))
        flat = refs.reshape(-1, refs.shape[-1]).astype(np.float64)
        global_rms = float(np.sqrt(np.mean(flat ** 2)))
        return {"metric": "action_parity", "value": float(cos.min()), "status": "measured",
                "parity_kind": "real_recorded_frames",
                "min_action_cosine": float(cos.min()), "mean_action_cosine": float(cos.mean()),
                "median_action_cosine": float(np.median(cos)),
                "per_frame_cosine": [float(x) for x in cosines], "n_obs": int(len(images)), "num_steps": num_steps,
                "max_per_dim_mae": max_pd, "mean_per_dim_mae": float(np.mean(per_dim_maes)),
                "per_dim_mae": [float(x) for x in per_dim], "action_rms": global_rms,
                "max_relative_action_mae": (max_pd / global_rms) if global_rms > 1e-9 else 0.0,
                "per_obs_ref_rms": per_obs_ref_rms,
                "dataset": str(d["dataset"]), "episode": int(d["episode"]),
                "observations": "REAL recorded SO-101 robot camera frames + state (LeRobot dataset), OOD rig "
                                "(up/side mapped to the policy's front/wrist) with fixed zero language tokens — "
                                "real pixels/state, not synthetic. Fidelity, not task success.",
                "reference": "torch fp16 encode+denoise vs the fp16 coreai-optimized asset, identical fixed-noise "
                             "Euler loop, on the SAME real recorded frames.",
                "runner": f"coreai-fabric-parity-runner/{__version__}", "environment": _environment()}

    result = asyncio.run(_run())
    (out / REAL_FRAME_PARITY_JSON).write_text(json.dumps(result, indent=2) + "\n")
    (out / ACTION_PARITY_JSON).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nok: wrote {out / REAL_FRAME_PARITY_JSON} and {out / ACTION_PARITY_JSON}")


def main():
    ap = argparse.ArgumentParser(description="SmolVLA real-recorded-frame action_parity")
    ap.add_argument("phase", nargs="?", default="reference", choices=["reference"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--dataset", default="lerobot/svla_so101_pickplace")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()
    if args.compare:
        cmd_compare(args.out, args.bundle or (args.out / "smolvla-so101.aimodel"))
    else:
        cmd_reference(args.out, args.dataset, args.episode, args.n_frames, args.seed, args.num_steps, fp16=not args.fp32)


if __name__ == "__main__":
    main()
