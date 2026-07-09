"""LingBot-Video (dense-1.3b / moe-30b-a3b) — VAE-decoder export + graph_output_cosine.

Apache-2.0 LingBotVideoPipeline (T2I/T2V/TI2V). The VAE is AutoencoderKLWan, config
IDENTICAL to lingbot-world-v2 (base_dim 96, z_dim 16, dim_mult [1,2,4,4],
temporal_downsample [f,t,t]) — this driver is the world-v2 VAE lane, parametrized by
--weights / --out so it serves both the dense and the MoE members.

Deployable graph = the VAE DECODER (pure conv, no attention): latent
[1,16,T,Hl,Wl] -> pixel frames. Single-chunk (first_chunk, num_frame=1) static
.aimodel, one program (no graph-split). Host owns the DiT denoise loop, latent
un-normalization (latents_mean/std), frame assembly, and the streaming feat_cache.
fp32. Gate B = graph_output_cosine (fp32 torch decoder vs .aimodel, seeded latents,
n_obs>=8). Unlike world-v2 (CC-BY-NC-SA), this family is Apache-2.0 -> publishable."""
import argparse, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric")
N_OBS, SEED = 8, 0
ZC, ZT, ZH, ZW = 16, 1, 60, 104  # latent tile -> [1,3,1,480,832]

VAE_CFG = dict(base_dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
               attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)


def build_vae(weights: Path):
    from diffusers import AutoencoderKLWan
    from safetensors.torch import load_file
    vae = AutoencoderKLWan(**VAE_CFG).eval()
    sd = load_file(str(weights))
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    print(f"loaded {len(sd)} tensors; missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    assert not missing, f"missing weights: {missing[:8]}"
    vae.use_tiling = False
    return vae


class Decoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__(); self.vae = vae

    def forward(self, z):
        return self.vae._decode(z, return_dict=False)[0]


def obs(i):
    g = torch.Generator().manual_seed(SEED + i)
    return (torch.randn(1, ZC, ZT, ZH, ZW, generator=g),)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="lingbot-video", help="reference label for the parity JSON")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    from coreai_torch import TorchConverter, get_decomp_table

    vae = build_vae(args.weights)
    dec = Decoder(vae).eval()
    obses = [obs(i) for i in range(N_OBS)]
    with torch.no_grad():
        refs = [dec(*o).float().numpy() for o in obses]
    print(f"torch fp32 ref OK, frames {refs[0].shape}", flush=True)
    if args.dry:
        print("DRY OK", flush=True); return

    with torch.no_grad():
        ep = torch.export.export(dec, args=obses[0], strict=False)
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(ep, input_names=["z"], output_names=["frames"],
                              entrypoint_name="vae_decode")
    prog = conv.to_coreai(); prog.optimize()
    args.out.mkdir(parents=True, exist_ok=True)
    aim = args.out / f"{args.out.name}.aimodel"; prog.save_asset(aim)
    sz = sum(f.stat().st_size for f in aim.rglob("*") if f.is_file())
    print(f"ok: saved fp32 {aim} (~{sz/1e6:.1f} MB)", flush=True)
    import gc; del prog, conv, ep; gc.collect()

    import asyncio
    from coreai.runtime import AIModel, NDArray
    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _environment

    async def run():
        mm = await AIModel.load(str(aim)); fn = mm.load_function("vae_decode")
        print("LOADS OK", flush=True)
        dt = np.dtype(str(fn.desc.input_descriptor("z").dtype))
        cos = []
        for o, r in zip(obses, refs):
            out = await fn(inputs={"z": NDArray(o[0].numpy().astype(dt))})
            a = out["frames"].numpy().astype(np.float64).reshape(-1)
            b = r.astype(np.float64).reshape(-1)
            cos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        cc = np.asarray(cos)
        res = {"metric": "graph_output_cosine", "value": float(cc.min()), "status": "measured",
               "min_cosine": float(cc.min()), "median_cosine": float(np.median(cc)),
               "mean_cosine": float(cc.mean()), "per_obs_cosine": [float(x) for x in cos],
               "n_obs": N_OBS, "reference_dtype": "float32", "quantization": "none",
               "asset_bytes": int(sz), "input_shape": [1, ZC, ZT, ZH, ZW],
               "output_shape": list(refs[0].shape),
               "runner": f"coreai-fabric-parity-runner/{__version__}", "environment": _environment(),
               "reference": (f"Torch AutoencoderKLWan first-chunk VAE decoder ({args.label}, fp32) "
                             "vs fp32 .aimodel over seeded latents. Single-chunk (num_frame=1).")}
        (args.out / "graph-output-parity-measured.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({k: res[k] for k in ("value", "min_cosine", "median_cosine", "n_obs")}, indent=2))
    asyncio.run(run())
    print("DONE", flush=True)


main()
