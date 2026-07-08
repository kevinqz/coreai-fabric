"""LingBot-VLA 2.0 — GRAPH-SPLIT action-denoise-step (loads on ANE).

The monolithic 36-layer dense-MoE graph exceeds the ANE per-program limit (0x10004).
This ships the action expert as N=3 layer-blocks of 12 layers each (fp16), plus a tiny
`embed` program and a `tail` folded into the last block. Each block is ~1.2GB — under
the ~1.5GB ceiling this graph hits (verified: fp16 L=12/14 load, L=18=1.79GB fails).

Host chains, loading each big block then FREEING it (the blocks must not be
co-resident in ANE memory):

  h, cond, cos, sin = embed(state, noisy_actions, timestep, position_ids)
  h = block_0(h, cond, cos, sin, prefix_k[0:12],  prefix_v[0:12],  attn_mask)   # free
  h = block_1(h, cond, cos, sin, prefix_k[12:24], prefix_v[12:24], attn_mask)   # free
  velocity = block_2(h, cond, cos, sin, prefix_k[24:36], prefix_v[24:36], attn_mask)  # + norm + action_out_proj

fp16 (better parity than int8; blocks are small enough to load). MoE dense-fused
(exact vs sparse top-4). Eager attention (no SDPA). Gate B = graph_output_cosine of
the CHAINED output vs the monolithic fp32 ref."""
import os, sys, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric")
sys.path.insert(0, str(ROOT / "models/lingbotvla"))
WEIGHTS = ROOT / "build/_lingbotvla/qwen_expert.safetensors"
OUT = ROOT / "build/lingbot-vla-v2"
N_OBS, SEED, CHUNK, PREFIX, NBLOCKS = 8, 0, 32, 32, 3
DRY = "--dry" in sys.argv
INT8 = "--int8" in sys.argv


def load_mapped(m):
    from safetensors.torch import load_file
    raw = load_file(str(WEIGHTS)); PRE = "qwenvl_with_expert.qwen_expert.model."
    sd = {}
    for k, v in raw.items():
        if k.startswith(PRE):
            sd[k[len(PRE):]] = v
        elif k.startswith(("action_in_proj", "action_out_proj", "action_time_mlp", "state_proj")):
            sd[k] = v
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"loaded {len(sd)} tensors; missing {len(miss)}, unexpected {len(unexp)}", flush=True)
    return miss, unexp


def main():
    from action_expert import LingbotActionExpert, LVConfig
    from coreai_torch import TorchConverter, get_decomp_table

    c = LVConfig(n_action_steps=CHUNK)
    m = LingbotActionExpert(c).eval()
    miss, unexp = load_mapped(m)
    assert not miss and not unexp, f"weight mismatch: {miss[:4]} / {unexp[:4]}"
    NL, KV, HD, SUF = c.num_layers, c.n_kv_heads, c.head_dim, CHUNK + 1
    L = NL // NBLOCKS
    bounds = [(i * L, (i + 1) * L if i < NBLOCKS - 1 else NL) for i in range(NBLOCKS)]  # last block absorbs remainder

    class Embed(torch.nn.Module):
        def __init__(s): super().__init__(); s.m = m
        def forward(s, state, noisy_actions, timestep, position_ids):
            cond, h = s.m.embed_suffix(state, noisy_actions, timestep)
            cos, sin = s.m._rope_cache(position_ids)
            return h, cond, cos, sin

    class BlockRange(torch.nn.Module):
        def __init__(s, lo, hi, tail): super().__init__(); s.m = m; s.lo, s.hi, s.tail = lo, hi, tail
        def forward(s, h, cond, cos, sin, prefix_k, prefix_v, attn_mask):
            for j in range(s.lo, s.hi):
                h = s.m.layers[j](h, cond, prefix_k[j - s.lo], prefix_v[j - s.lo], cos, sin, attn_mask)
            if s.tail:
                h = s.m.norm(h)
                return s.m.action_out_proj(h[:, -c.n_action_steps:])
            return h

    def obs(i):
        g = torch.Generator().manual_seed(SEED + i)
        return dict(state=torch.randn(1, c.state_dim, generator=g), na=torch.randn(1, CHUNK, c.action_dim, generator=g),
                    ts=torch.rand(1, generator=g), pk=torch.randn(NL, 1, PREFIX, KV, HD, generator=g),
                    pv=torch.randn(NL, 1, PREFIX, KV, HD, generator=g),
                    pos=torch.arange(PREFIX, PREFIX + SUF).view(1, SUF), mask=torch.ones(1, SUF, PREFIX + SUF, dtype=torch.bool))

    obses = [obs(i) for i in range(N_OBS)]
    with torch.no_grad():
        refs = [m.forward(o["state"], o["na"], o["ts"], o["pk"], o["pv"], o["pos"], o["mask"]).float().numpy() for o in obses]
    print(f"fp32 monolithic ref OK {refs[0].shape}; blocks={bounds}", flush=True)
    if DRY:
        print("DRY OK", flush=True); return

    if INT8:
        m.quantize_experts_int8()
    EM = Embed().half().eval()
    blocks = [BlockRange(lo, hi, tail=(i == NBLOCKS - 1)).half().eval() for i, (lo, hi) in enumerate(bounds)]

    def hf(o, k):
        t = o[k]; return t.half() if t.is_floating_point() else t
    o0 = obses[0]

    OUT.mkdir(parents=True, exist_ok=True)
    aims = {}
    # embed
    aE = (hf(o0, "state"), hf(o0, "na"), hf(o0, "ts"), o0["pos"])
    with torch.no_grad():
        hcur, cond, cos, sin = EM(*aE)
        epE = torch.export.export(EM, args=aE, strict=False).run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(epE, input_names=["state", "noisy_actions", "timestep", "position_ids"],
                              output_names=["h", "cond", "cos", "sin"], entrypoint_name="embed")
    p = conv.to_coreai(); p.optimize(); a = OUT / "lingbot-vla-v2-embed.aimodel"; p.save_asset(a); aims["embed"] = a
    print(f"  saved embed (~{sum(f.stat().st_size for f in a.rglob('*') if f.is_file())/1e9:.2f}GB)", flush=True)
    del p, conv, epE
    # blocks
    for i, (lo, hi) in enumerate(bounds):
        name = f"block{i}"
        aB = (hcur, cond, cos, sin, hf(o0, "pk")[lo:hi], hf(o0, "pv")[lo:hi], o0["mask"])
        with torch.no_grad():
            out = blocks[i](*aB)
            epB = torch.export.export(blocks[i], args=aB, strict=False).run_decompositions(get_decomp_table())
        conv = TorchConverter()
        outn = ["velocity"] if i == NBLOCKS - 1 else ["h_out"]
        conv.add_exported_program(epB, input_names=["h", "cond", "cos", "sin", "prefix_k", "prefix_v", "attn_mask"],
                                  output_names=outn, entrypoint_name=name)
        p = conv.to_coreai(); p.optimize(); a = OUT / f"lingbot-vla-v2-{name}.aimodel"; p.save_asset(a); aims[name] = a
        print(f"  saved {name} layers[{lo}:{hi}] (~{sum(f.stat().st_size for f in a.rglob('*') if f.is_file())/1e9:.2f}GB)", flush=True)
        hcur = out  # thread the example forward for the next block's export shapes
        del p, conv, epB
    import gc; gc.collect()
    sz = sum(f.stat().st_size for a in aims.values() for f in a.rglob("*") if f.is_file())

    import asyncio
    from coreai.runtime import AIModel, NDArray
    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _environment

    def nd(fn, n, arr):
        dt = np.dtype(str(fn.desc.input_descriptor(n).dtype)); return NDArray(np.asarray(arr).astype(dt))

    async def run():
        import gc as _gc
        # embed (tiny) — run all obs, keep intermediates
        me = await AIModel.load(str(aims["embed"])); fe = me.load_function("embed")
        inter = []  # per-obs (h, cond, cos, sin)
        for o in obses:
            oe = await fe(inputs={"state": nd(fe, "state", hf(o, "state").numpy()), "noisy_actions": nd(fe, "noisy_actions", hf(o, "na").numpy()),
                                  "timestep": nd(fe, "timestep", hf(o, "ts").numpy()), "position_ids": nd(fe, "position_ids", o["pos"].numpy())})
            inter.append([oe["h"].numpy(), oe["cond"].numpy(), oe["cos"].numpy(), oe["sin"].numpy()])
        print("embed: ran", len(obses), "obs", flush=True)
        del me, fe; _gc.collect()
        # blocks, sequential load -> run -> free (never co-resident)
        velos = [None] * N_OBS
        for i, (lo, hi) in enumerate(bounds):
            name = f"block{i}"
            mb = await AIModel.load(str(aims[name])); fb = mb.load_function(name)
            print(f"{name}: LOADS OK", flush=True)
            outn = "velocity" if i == NBLOCKS - 1 else "h_out"
            for oi, o in enumerate(obses):
                hh, cond, cos, sin = inter[oi]
                rb = await fb(inputs={"h": nd(fb, "h", hh), "cond": nd(fb, "cond", cond), "cos": nd(fb, "cos", cos), "sin": nd(fb, "sin", sin),
                                      "prefix_k": nd(fb, "prefix_k", hf(o, "pk")[lo:hi].numpy()), "prefix_v": nd(fb, "prefix_v", hf(o, "pv")[lo:hi].numpy()),
                                      "attn_mask": nd(fb, "attn_mask", o["mask"].numpy())})
                if i == NBLOCKS - 1:
                    velos[oi] = rb[outn].numpy()
                else:
                    inter[oi][0] = rb[outn].numpy()
            del mb, fb; _gc.collect()
        cos_scores = []
        for v, r in zip(velos, refs):
            aa = v.astype(np.float64).reshape(-1); bb = r.astype(np.float64).reshape(-1)
            cos_scores.append(float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12)))
        cc = np.asarray(cos_scores)
        res = {"metric": "graph_output_cosine", "value": float(cc.min()), "status": "measured", "min_cosine": float(cc.min()),
               "median_cosine": float(np.median(cc)), "mean_cosine": float(cc.mean()), "per_obs_cosine": [float(x) for x in cos_scores],
               "n_obs": N_OBS, "reference_dtype": "float32", "quantization": ("float16+int8_experts" if INT8 else "float16"), "asset_bytes": int(sz),
               "graphs": ["embed"] + [f"block{i} (layers {lo}-{hi-1}{' + tail' if i == NBLOCKS-1 else ''})" for i, (lo, hi) in enumerate(bounds)],
               "runner": f"coreai-fabric-parity-runner/{__version__}", "environment": _environment(),
               "reference": f"Torch LingBot-VLA-2.0 MoE monolithic fp32 denoise-step velocity vs the CHAINED {NBLOCKS}-block split (embed -> block0..block{NBLOCKS-1}) over seeded inputs; blocks loaded sequentially (never co-resident). MoE dense-fused (exact vs sparse top-4)."}
        (OUT / "graph-output-parity-measured.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({k: res[k] for k in ("value", "min_cosine", "median_cosine", "n_obs")}, indent=2))
    asyncio.run(run())
    print("DONE", flush=True)


main()
