# /// script
# requires-python = ">=3.12"
# ///
"""Diffusion Policy (lerobot) -> Apple .aimodel, SPLIT export (encode + denoise_step).

This is the multi-step SAMPLER lane — the tiny, VLM-free analog of pi0's
flow-matching loop, used to prove the split-export + host-driven sampler pattern
end-to-end before pi0. A Diffusion Policy is:
  encode:       (images, state) over n_obs steps -> global_cond   (run ONCE)
  denoise_step: (x_t, timestep, global_cond)     -> eps           (host drives N times)
The N-step DDPM update + un-normalization live in host code (net-new, mirrors pi0).
Both graphs land in ONE .aimodel as named entrypoints (encode / denoise_step), the
exact contract coreai-fabric-parity-runner's action_parity compare expects.

TWO-VENV (like act/export.py): export in venv-A (.venv-lerobot, torch 2.9 +
lerobot[pi]) -> encode.pt2 + denoise.pt2 + sampler.json; --lower in venv-B (fabric
.venv, coreai_torch) -> the .aimodel. See docs/vla-export-runbook.md.

Usage:
  .venv-lerobot/bin/python models/diffusion/export.py export --repo lerobot/diffusion_pusht --out build/diffusion-pusht
  .venv/bin/python         models/diffusion/export.py --lower  --out build/diffusion-pusht
  # op-coverage probe (denoise only — the real unknown): add --only denoise
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OBS_STATE = "observation.state"
OBS_IMAGES = "observation.images"


def _build(repo: str):
    """venv-A: load the Diffusion Policy on CPU + return (diffusion_model, shapes)."""
    import torch
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    policy = DiffusionPolicy.from_pretrained(repo).to("cpu").eval()
    dm = policy.diffusion            # DiffusionModel: .unet, .rgb_encoder, .config, .noise_scheduler
    cfg = dm.config
    n_obs = int(cfg.n_obs_steps)
    horizon = int(cfg.horizon)
    action_dim = int(cfg.action_feature.shape[0])
    img_key = next((k for k, f in cfg.input_features.items() if f.type == "VISUAL"), None)
    img_shape = tuple(cfg.input_features[img_key].shape) if img_key else None   # (3,96,96)
    state_dim = int(cfg.input_features[OBS_STATE].shape[0])
    num_cams = len(cfg.image_features) if cfg.image_features else 0

    # global_cond dim: (state + per-cam feature) * n_obs  (see DiffusionModel.__init__)
    feat = dm.rgb_encoder.feature_dim if hasattr(dm, "rgb_encoder") else 0
    global_cond_dim = (state_dim + feat * num_cams) * n_obs

    shapes = {
        "n_obs": n_obs, "horizon": horizon, "action_dim": action_dim,
        "state_dim": state_dim, "num_cams": num_cams, "img_shape": list(img_shape) if img_shape else None,
        "global_cond_dim": int(global_cond_dim),
        "num_inference_steps": int(dm.num_inference_steps),
        "num_train_timesteps": int(cfg.num_train_timesteps),
        "prediction_type": cfg.prediction_type,
        "clip_sample": bool(cfg.clip_sample),
        "clip_sample_range": float(cfg.clip_sample_range),
        "n_action_steps": int(cfg.n_action_steps),
    }
    # Sampler constants for the host DDPM loop: the exact alphas_cumprod + timesteps
    # the reference scheduler uses (so the host loop reproduces it bit-for-bit).
    sched = dm.noise_scheduler
    sched.set_timesteps(dm.num_inference_steps)
    shapes["timesteps"] = [int(t) for t in sched.timesteps.tolist()]
    shapes["alphas_cumprod"] = [float(a) for a in sched.alphas_cumprod.tolist()]
    return dm, shapes


def _wrappers(dm, shapes):
    import torch

    class EncodeWrap(torch.nn.Module):
        def __init__(self, dm): super().__init__(); self.dm = dm
        def forward(self, images, state):
            return self.dm._prepare_global_conditioning({OBS_IMAGES: images, OBS_STATE: state})

    class DenoiseWrap(torch.nn.Module):
        def __init__(self, dm): super().__init__(); self.unet = dm.unet
        def forward(self, x_t, timestep, global_cond):
            return self.unet(x_t, timestep, global_cond=global_cond)

    return EncodeWrap(dm).eval(), DenoiseWrap(dm).eval()


def _example_inputs(shapes):
    import torch
    n_obs, horizon, adim = shapes["n_obs"], shapes["horizon"], shapes["action_dim"]
    C, H, W = shapes["img_shape"]
    ncam, sdim, gdim = shapes["num_cams"], shapes["state_dim"], shapes["global_cond_dim"]
    images = torch.zeros(1, n_obs, ncam, C, H, W)
    state = torch.zeros(1, n_obs, sdim)
    x_t = torch.zeros(1, horizon, adim)
    timestep = torch.zeros(1, dtype=torch.long)
    global_cond = torch.zeros(1, gdim)
    return images, state, x_t, timestep, global_cond


def cmd_export(repo: str, out: Path, only: str | None):
    import torch
    dm, shapes = _build(repo)
    enc, den = _wrappers(dm, shapes)
    images, state, x_t, timestep, global_cond = _example_inputs(shapes)
    out.mkdir(parents=True, exist_ok=True)

    if only in (None, "denoise"):
        den_ep = torch.export.export(den, args=(x_t, timestep, global_cond), strict=False)
        torch.export.save(den_ep, str(out / "denoise.pt2"))
        print(f"ok: wrote {out}/denoise.pt2  (x_t{list(x_t.shape)} t{list(timestep.shape)} "
              f"cond{list(global_cond.shape)} -> eps{list(x_t.shape)})")
    if only in (None, "encode"):
        enc_ep = torch.export.export(enc, args=(images, state), strict=False)
        torch.export.save(enc_ep, str(out / "encode.pt2"))
        print(f"ok: wrote {out}/encode.pt2  (images{list(images.shape)} state{list(state.shape)} "
              f"-> global_cond[1,{shapes['global_cond_dim']}])")

    (out / "sampler.json").write_text(json.dumps(shapes, indent=2))
    print(f"ok: wrote {out}/sampler.json ({shapes['num_inference_steps']}-step "
          f"{shapes['prediction_type']} DDPM, horizon {shapes['horizon']}x{shapes['action_dim']})")
    print("next (venv-B): .venv/bin/python models/diffusion/export.py --lower --out", out)


def cmd_lower(out: Path, only: str | None):
    """venv-B: load the .pt2(s) -> one .aimodel with encode + denoise_step entrypoints."""
    import torch
    from coreai_torch import TorchConverter, get_decomp_table

    def load(name):
        ep = torch.export.load(str(out / f"{name}.pt2"))
        return ep.run_decompositions(get_decomp_table())

    conv = TorchConverter()
    added = []
    if only in (None, "denoise") and (out / "denoise.pt2").exists():
        conv.add_exported_program(load("denoise"), input_names=["x_t", "timestep", "global_cond"],
                                  output_names=["eps"], entrypoint_name="denoise_step")
        added.append("denoise_step")
    if only in (None, "encode") and (out / "encode.pt2").exists():
        conv.add_exported_program(load("encode"), input_names=["images", "state"],
                                  output_names=["global_cond"], entrypoint_name="encode")
        added.append("encode")

    prog = conv.to_coreai()
    prog.optimize()
    aimodel = out / f"{out.name}.aimodel"
    prog.save_asset(aimodel)
    print(f"ok: lowered Diffusion Policy -> {aimodel}  (entrypoints: {', '.join(added)})")


def main():
    ap = argparse.ArgumentParser(description="Diffusion Policy split export (see docs/vla-export-runbook.md)")
    ap.add_argument("phase", nargs="?", default="export", choices=["export"])
    ap.add_argument("--repo", default="lerobot/diffusion_pusht")
    ap.add_argument("--lower", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", choices=["encode", "denoise"], default=None,
                    help="export/lower only one graph (op-coverage probe)")
    args = ap.parse_args()
    if args.lower:
        cmd_lower(args.out, args.only)
    else:
        cmd_export(args.repo, args.out, args.only)


if __name__ == "__main__":
    main()
