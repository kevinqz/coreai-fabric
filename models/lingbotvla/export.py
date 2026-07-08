"""LingBot-VLA 2.0 action-denoise-step — int8 export + graph_output_cosine parity.

Deployable graph = the 36-layer MoE action expert doing one flow-matching denoise step,
conditioned on the Qwen3-VL prefix K/V (fed as prefix_k/prefix_v graph inputs). Host
owns embed_prefix (Qwen3-VL), the Euler loop, un-normalization. MoE dense-fused (exact
vs sparse top-4), eager attention (no SDPA). int8 weight-only (torchao) — the ~1.8B
expert is ~7.15GB fp32; int8 makes it ~1.8GB for the ANE."""
import os, sys, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric")
sys.path.insert(0, str(ROOT / "models/lingbotvla"))
WEIGHTS = ROOT / "build/_lingbotvla/qwen_expert.safetensors"
OUT = ROOT / "build/lingbot-vla-v2"
N_OBS, SEED = 8, 0
CHUNK, PREFIX = 32, 32
DRY = "--dry" in sys.argv


def load_mapped(m):
    from safetensors.torch import load_file
    raw = load_file(str(WEIGHTS))
    PRE = "qwenvl_with_expert.qwen_expert.model."
    sd = {}
    for k, v in raw.items():
        if k.startswith(PRE):
            sd[k[len(PRE):]] = v
        elif k.startswith(("action_in_proj", "action_out_proj", "action_time_mlp", "state_proj")):
            sd[k] = v
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"loaded {len(sd)} tensors; missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    if missing:
        print("  MISSING:", missing[:8], flush=True)
    if unexpected:
        print("  UNEXPECTED:", unexpected[:8], flush=True)


def main():
    from action_expert import LingbotActionExpert, LVConfig
    from coreai_torch import TorchConverter, get_decomp_table

    c = LVConfig(n_action_steps=CHUNK)
    m = LingbotActionExpert(c).eval()
    load_mapped(m)
    NL, KV, HD = c.num_layers, c.n_kv_heads, c.head_dim
    SUF = CHUNK + 1

    class Step(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.m = m

        def forward(s, state, noisy_actions, timestep, prefix_k, prefix_v, position_ids, attn_mask):
            return s.m.forward(state, noisy_actions, timestep, prefix_k, prefix_v, position_ids, attn_mask)

    step = Step().eval()

    def obs(i):
        g = torch.Generator().manual_seed(SEED + i)
        state = torch.randn(1, c.state_dim, generator=g)
        na = torch.randn(1, CHUNK, c.action_dim, generator=g)
        ts = torch.rand(1, generator=g)
        pk = torch.randn(NL, 1, PREFIX, KV, HD, generator=g)
        pv = torch.randn(NL, 1, PREFIX, KV, HD, generator=g)
        pos = torch.arange(PREFIX, PREFIX + SUF).view(1, SUF)          # suffix positions
        mask = torch.ones(1, SUF, PREFIX + SUF, dtype=torch.bool)      # host-built; all-valid for parity
        return state, na, ts, pk, pv, pos, mask

    obses = [obs(i) for i in range(N_OBS)]
    with torch.no_grad():
        refs = [step(*o).float().numpy() for o in obses]
    print("torch fp32 ref OK, velocity", refs[0].shape, flush=True)
    if DRY:
        print("DRY OK — build+load+forward validated", flush=True); return

    m.quantize_experts_int8()   # per-channel int8 on the 3D experts (bulk); torchao can't einsum them
    step = step.half().eval()   # fp16 for the rest; int8 experts + fp16 -> ~1.9GB, loads on ANE
    obses = [tuple(t.half() if t.is_floating_point() else t for t in o) for o in obses]
    refs8 = refs   # fp32 refs; parity = (int8-experts + fp16) asset vs fp32 ref
    with torch.no_grad():
        ep = torch.export.export(step, args=obses[0], strict=False)
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(ep, input_names=["state", "noisy_actions", "timestep", "prefix_k", "prefix_v", "position_ids", "attn_mask"],
                              output_names=["velocity"], entrypoint_name="action_denoise_step")
    prog = conv.to_coreai(); prog.optimize()
    OUT.mkdir(parents=True, exist_ok=True)
    aim = OUT / f"{OUT.name}.aimodel"; prog.save_asset(aim)
    sz = sum(f.stat().st_size for f in aim.rglob("*") if f.is_file())
    print(f"ok: saved fp16 {aim} (~{sz/1e9:.2f} GB)", flush=True)
    import gc; del prog, conv, ep; gc.collect()

    import asyncio
    from coreai.runtime import AIModel, NDArray
    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _environment
    NAMES = ["state", "noisy_actions", "timestep", "prefix_k", "prefix_v", "position_ids", "attn_mask"]

    async def run():
        mm = await AIModel.load(str(aim)); fn = mm.load_function("action_denoise_step")
        print("LOADS OK", flush=True)

        def nd(n, a):
            dt = np.dtype(str(fn.desc.input_descriptor(n).dtype)); return NDArray(np.asarray(a).astype(dt))
        cos = []
        for o, r in zip(obses, refs8):
            ins = {nm: nd(nm, t.numpy()) for nm, t in zip(NAMES, o)}
            out = await fn(inputs=ins)
            a = out["velocity"].numpy().astype(np.float64).reshape(-1); b = r.astype(np.float64).reshape(-1)
            cos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        cc = np.asarray(cos)
        res = {"metric": "graph_output_cosine", "value": float(cc.min()), "status": "measured", "min_cosine": float(cc.min()),
               "median_cosine": float(np.median(cc)), "mean_cosine": float(cc.mean()), "per_obs_cosine": [float(x) for x in cos],
               "n_obs": N_OBS, "reference_dtype": "float32", "quantization": "float16", "asset_bytes": int(sz),
               "runner": f"coreai-fabric-parity-runner/{__version__}", "environment": _environment(),
               "reference": "Torch LingBot-VLA-2.0 MoE action-expert denoise-step velocity vs int8 .aimodel over seeded inputs (host owns Qwen3-VL prefix K/V + Euler loop). MoE dense-fused (exact vs sparse top-4)."}
        (OUT / "graph-output-parity-measured.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({k: res[k] for k in ("value", "min_cosine", "n_obs")}, indent=2))
    asyncio.run(run())
    print("DONE", flush=True)


main()
