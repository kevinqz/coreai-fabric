"""LingBot-World V2 14B causal-fast — VAE-decoder export + graph_output_cosine parity.

Deployable graph = the AutoencoderKLWan VIDEO VAE decoder (the proven encoder-lane
sibling: pure conv, NO attention). Input latent [1,16,T,Hl,Wl] -> pixel frames
[1,3,Tp,Hl*8,Wl*8]. The host owns the DiT few-step denoise loop, the latent
un-normalization (latents_mean / latents_std, 16 channels), and frame assembly.

SINGLE-CHUNK: this ships the `first_chunk` decode (num_frame=1, use_tiling off) as one
static-size .aimodel — small enough to sidestep the 0x10004 ceiling (no graph-split).
Streaming multi-chunk decode needs the WAN causal feat_cache threaded as graph I/O
(same technique as the VLA prefix-K/V) — that is the documented follow-up.

fp32 (recipe: quantization none / precision float32). Gate B = graph_output_cosine of
the fp32 torch decoder vs the .aimodel over seeded latents (min cosine, n_obs>=8)."""
import os, sys, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric")
WEIGHTS = ROOT / "build/_lingbotworld/vae.safetensors"
OUT = ROOT / "build/lingbot-world-v2-14b-causal-fast"
N_OBS, SEED = 8, 0
# latent tile -> pixels: [1,16,1,60,104] decodes to [1,3,1,480,832] (the causal-fast res)
ZC, ZT, ZH, ZW = 16, 1, 60, 104
DRY = "--dry" in sys.argv

VAE_CFG = dict(base_dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
               attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)


def build_vae():
    from diffusers import AutoencoderKLWan
    from safetensors.torch import load_file
    vae = AutoencoderKLWan(**VAE_CFG).eval()
    sd = load_file(str(WEIGHTS))
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    print(f"loaded {len(sd)} tensors; missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    if missing:
        print("  MISSING:", missing[:8], flush=True)
    if unexpected:
        print("  UNEXPECTED:", unexpected[:8], flush=True)
    vae.use_tiling = False
    return vae


class Decoder(torch.nn.Module):
    """latent -> pixels, first-chunk causal decode (num_frame=1)."""
    def __init__(self, vae):
        super().__init__(); self.vae = vae

    def forward(self, z):
        return self.vae._decode(z, return_dict=False)[0]


def obs(i):
    g = torch.Generator().manual_seed(SEED + i)
    return (torch.randn(1, ZC, ZT, ZH, ZW, generator=g),)


def main():
    from coreai_torch import TorchConverter, get_decomp_table

    vae = build_vae()
    dec = Decoder(vae).eval()
    obses = [obs(i) for i in range(N_OBS)]
    with torch.no_grad():
        refs = [dec(*o).float().numpy() for o in obses]
    print(f"torch fp32 ref OK, frames {refs[0].shape}", flush=True)
    if DRY:
        print("DRY OK — build+decode validated", flush=True); return

    with torch.no_grad():
        ep = torch.export.export(dec, args=obses[0], strict=False)
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(ep, input_names=["z"], output_names=["frames"],
                              entrypoint_name="vae_decode")
    prog = conv.to_coreai(); prog.optimize()
    OUT.mkdir(parents=True, exist_ok=True)
    aim = OUT / f"{OUT.name}.aimodel"; prog.save_asset(aim)
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
               "reference": ("Torch AutoencoderKLWan first-chunk VAE decoder (latent->pixels, fp32) "
                             "vs fp32 .aimodel over seeded latents. Host owns DiT denoise loop, latent "
                             "un-normalization, and streaming feat_cache. Single-chunk (num_frame=1).")}
        (OUT / "graph-output-parity-measured.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({k: res[k] for k in ("value", "min_cosine", "median_cosine", "n_obs")}, indent=2))
    asyncio.run(run())
    print("DONE", flush=True)


main()
