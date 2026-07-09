"""LingBot-Video VAE decoder — STREAMING (subsequent-chunk) export + graph_output_cosine.

The published VAE asset does the WAN first_chunk decode. Continuous video needs the causal
conv cache (feat_cache) threaded across chunks. This ships the SUBSEQUENT-chunk graph:
(latent_chunk [1,16,1,Hl,Wl], 32 cache_in tensors) -> (frames, 32 cache_out tensors),
first_chunk=False. The host decodes chunk 0 with the published first-chunk asset, then loops
THIS asset feeding cache_out -> cache_in. 32 cache tensors (6 distinct shapes, ~155MB fp32);
feat_map slot 0 is un-cached (None). fp32. Gate B = graph_output_cosine of the streaming
.aimodel vs the torch decoder subsequent-chunk output over seeded (chunk, cache) states."""
import sys, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric")
WEIGHTS = ROOT / "build/_lingbotvideo/dense_vae.safetensors"
OUT = ROOT / "build/lingbot-video-dense-1.3b-vae-streaming"
N_OBS, SEED = 8, 0
ZC, ZH, ZW = 16, 16, 16
VAE_CFG = dict(base_dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
               attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)
DRY = "--dry" in sys.argv


def build_vae():
    from diffusers import AutoencoderKLWan
    from safetensors.torch import load_file
    vae = AutoencoderKLWan(**VAE_CFG).eval()
    vae.load_state_dict(load_file(str(WEIGHTS)), strict=False)
    vae.use_tiling = False
    return vae


def prime_cache(vae, z_prev):
    """Decode chunk0 (first_chunk) + chunk1 (subsequent) so the cache reaches STEADY STATE
    (all tensors, no 'Rep' markers). z_prev: [1,16,2,Hl,Wl]. Returns the 33-slot cache list."""
    vae.clear_cache()
    x = vae.post_quant_conv(z_prev)
    vae._conv_idx = [0]
    vae.decoder(x[:, :, 0:1], feat_cache=vae._feat_map, feat_idx=vae._conv_idx, first_chunk=True)
    vae._conv_idx = [0]
    vae.decoder(x[:, :, 1:2], feat_cache=vae._feat_map, feat_idx=vae._conv_idx)  # -> steady-state tensors
    return list(vae._feat_map)


class StreamStep(torch.nn.Module):
    """Subsequent-chunk decode: (z_chunk, *cache_in) -> (frames, *cache_out). Only the TENSOR
    cache slots (tensor_idx) are threaded as I/O; non-tensor slots (None) are baked from the
    template (they don't carry cross-chunk state)."""
    def __init__(s, vae, n_slots, tensor_idx):
        super().__init__(); s.vae = vae; s.n_slots = n_slots; s.tensor_idx = list(tensor_idx)
    def forward(s, z_chunk, *cache_in):
        x = s.vae.post_quant_conv(z_chunk)
        fm = [None] * s.n_slots
        for j, i in enumerate(s.tensor_idx):
            fm[i] = cache_in[j]
        s.vae._feat_map = fm
        s.vae._conv_idx = [0]
        out = s.vae.decoder(x[:, :, 0:1], feat_cache=s.vae._feat_map, feat_idx=s.vae._conv_idx)
        cache_out = tuple(s.vae._feat_map[i] for i in s.tensor_idx)
        return (out, *cache_out)


def main():
    from coreai_torch import TorchConverter, get_decomp_table
    vae = build_vae()
    # template cache from priming -> which slots carry tensor state (threaded as I/O)
    tmpl = prime_cache(vae, torch.randn(1, ZC, 2, ZH, ZW))
    n_slots = len(tmpl)
    tensor_idx = [i for i, t in enumerate(tmpl) if torch.is_tensor(t)]
    print(f"cache slots: {n_slots} | tensor slots (threaded): {len(tensor_idx)} | baked-None: {n_slots-len(tensor_idx)}", flush=True)

    def make(i):
        g = torch.Generator().manual_seed(SEED + i)
        z_prev = torch.randn(1, ZC, 2, ZH, ZW, generator=g)
        z_chunk = torch.randn(1, ZC, 1, ZH, ZW, generator=g)
        cache_full = prime_cache(vae, z_prev)
        cache = [cache_full[i].detach().clone() for i in tensor_idx]
        return z_chunk, cache
    obses = [make(i) for i in range(N_OBS)]
    n_cache = len(tensor_idx)
    step = StreamStep(vae, n_slots, tensor_idx).eval()
    with torch.no_grad():
        refs = [step(o[0], *o[1])[0].float().numpy() for o in obses]
    print("torch subsequent-chunk ref OK, frames", refs[0].shape, flush=True)
    if DRY:
        print("DRY OK", flush=True); return

    args0 = (obses[0][0], *obses[0][1])
    in_names = ["z_chunk"] + [f"cache_in_{i}" for i in range(n_cache)]
    out_names = ["frames"] + [f"cache_out_{i}" for i in range(n_cache)]
    with torch.no_grad():
        ep = torch.export.export(step, args=args0, strict=False)
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(ep, input_names=in_names, output_names=out_names,
                              entrypoint_name="vae_decode_stream")
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
        mm = await AIModel.load(str(aim)); fn = mm.load_function("vae_decode_stream")
        print("LOADS OK", flush=True)
        def nd(n, a):
            dt = np.dtype(str(fn.desc.input_descriptor(n).dtype)); return NDArray(np.asarray(a).astype(dt))
        cos = []
        for o, r in zip(obses, refs):
            ins = {"z_chunk": nd("z_chunk", o[0].numpy())}
            for i, t in enumerate(o[1]):
                ins[f"cache_in_{i}"] = nd(f"cache_in_{i}", t.numpy())
            out = await fn(inputs=ins)
            a = out["frames"].numpy().astype(np.float64).reshape(-1); b = r.astype(np.float64).reshape(-1)
            cos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        cc = np.asarray(cos)
        res = {"metric": "graph_output_cosine", "value": float(cc.min()), "status": "measured",
               "min_cosine": float(cc.min()), "median_cosine": float(np.median(cc)),
               "mean_cosine": float(cc.mean()), "per_obs_cosine": [float(x) for x in cos],
               "n_obs": N_OBS, "reference_dtype": "float32", "quantization": "none", "asset_bytes": int(sz),
               "output_shape": list(refs[0].shape), "n_cache_tensors": n_cache,
               "runner": f"coreai-fabric-parity-runner/{__version__}", "environment": _environment(),
               "reference": "Torch AutoencoderKLWan subsequent-chunk decode (feat_cache threaded as 32-tensor graph I/O) vs the .aimodel over seeded (chunk, primed-cache) states."}
        (OUT / "graph-output-parity-measured.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({k: res[k] for k in ("value", "min_cosine", "median_cosine", "n_obs", "n_cache_tensors")}, indent=2))
    asyncio.run(run())
    print("DONE", flush=True)


main()
