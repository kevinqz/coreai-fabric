"""Validate the 2-stage split mechanics: one asset, two entrypoints (stage_a/stage_b),
host chains A->B. Tiny random config. Checks the chained output == monolithic torch fp16
(same weights) so the split wiring (residual/cond/cos/sin hand-off) is correct."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "/Users/kevinsaltarelli/Dev/Github/coreai-fabric/models/lingbotvla")
from action_expert import LingbotActionExpert, LVConfig
from coreai_torch import TorchConverter, get_decomp_table

CHUNK, PREFIX, NL, SPLIT = 4, 4, 4, 2
c = LVConfig(num_layers=NL, n_action_steps=CHUNK)
m = LingbotActionExpert(c).eval()
for l in m.layers:
    for p in (l.mlp.experts.gate_proj, l.mlp.experts.up_proj, l.mlp.experts.down_proj):
        torch.nn.init.normal_(p, std=0.02)
KV, HD, SUF = c.n_kv_heads, c.head_dim, CHUNK + 1
g = torch.Generator().manual_seed(0)
state = torch.randn(1, c.state_dim, generator=g); na = torch.randn(1, CHUNK, c.action_dim, generator=g)
ts = torch.rand(1, generator=g); pk = torch.randn(NL, 1, PREFIX, KV, HD, generator=g); pv = torch.randn(NL, 1, PREFIX, KV, HD, generator=g)
pos = torch.arange(PREFIX, PREFIX + SUF).view(1, SUF); mask = torch.ones(1, SUF, PREFIX + SUF, dtype=torch.bool)

class StageA(torch.nn.Module):
    def __init__(s): super().__init__(); s.m = m
    def forward(s, state, na, ts, pos, pk, pv, mask):
        cond, h = s.m.embed_suffix(state, na, ts); cos, sin = s.m._rope_cache(pos)
        for i in range(SPLIT): h = s.m.layers[i](h, cond, pk[i], pv[i], cos, sin, mask)
        return h, cond, cos, sin
class StageB(torch.nn.Module):
    def __init__(s): super().__init__(); s.m = m
    def forward(s, h, cond, cos, sin, pk, pv, mask):
        for j in range(SPLIT, NL): h = s.m.layers[j](h, cond, pk[j-SPLIT], pv[j-SPLIT], cos, sin, mask)
        return s.m.action_out_proj(s.m.norm(h)[:, -c.n_action_steps:])

with torch.no_grad(): ref = m.forward(state, na, ts, pk, pv, pos, mask).float().numpy()
A = StageA().half().eval(); B = StageB().half().eval()
hf = lambda t: t.half() if t.is_floating_point() else t
aA = (hf(state), hf(na), hf(ts), pos, hf(pk)[:SPLIT], hf(pv)[:SPLIT], mask)
with torch.no_grad():
    hh, cond, cos, sin = A(*aA)
    epA = torch.export.export(A, args=aA, strict=False).run_decompositions(get_decomp_table())
    aB = (hh, cond, cos, sin, hf(pk)[SPLIT:], hf(pv)[SPLIT:], mask)
    epB = torch.export.export(B, args=aB, strict=False).run_decompositions(get_decomp_table())
conv = TorchConverter()
conv.add_exported_program(epA, input_names=["state","noisy_actions","timestep","position_ids","prefix_k","prefix_v","attn_mask"],
                          output_names=["h","cond","cos","sin"], entrypoint_name="stage_a")
conv.add_exported_program(epB, input_names=["h","cond","cos","sin","prefix_k","prefix_v","attn_mask"],
                          output_names=["velocity"], entrypoint_name="stage_b")
prog = conv.to_coreai(); prog.optimize()
aim = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric/build/_splsmoke/s.aimodel")
aim.parent.mkdir(parents=True, exist_ok=True); prog.save_asset(aim)
print("saved 2-entrypoint asset", flush=True)
import asyncio
from coreai.runtime import AIModel, NDArray
async def run():
    mm = await AIModel.load(str(aim)); fa = mm.load_function("stage_a"); fb = mm.load_function("stage_b")
    print("BOTH FUNCTIONS LOAD OK", flush=True)
    def nd(fn,n,a): dt=np.dtype(str(fn.desc.input_descriptor(n).dtype)); return NDArray(np.asarray(a).astype(dt))
    oa = await fa(inputs={"state":nd(fa,"state",hf(state).numpy()),"noisy_actions":nd(fa,"noisy_actions",hf(na).numpy()),
                          "timestep":nd(fa,"timestep",hf(ts).numpy()),"position_ids":nd(fa,"position_ids",pos.numpy()),
                          "prefix_k":nd(fa,"prefix_k",hf(pk)[:SPLIT].numpy()),"prefix_v":nd(fa,"prefix_v",hf(pv)[:SPLIT].numpy()),
                          "attn_mask":nd(fa,"attn_mask",mask.numpy())})
    ob = await fb(inputs={"h":nd(fb,"h",oa["h"].numpy()),"cond":nd(fb,"cond",oa["cond"].numpy()),
                          "cos":nd(fb,"cos",oa["cos"].numpy()),"sin":nd(fb,"sin",oa["sin"].numpy()),
                          "prefix_k":nd(fb,"prefix_k",hf(pk)[SPLIT:].numpy()),"prefix_v":nd(fb,"prefix_v",hf(pv)[SPLIT:].numpy()),
                          "attn_mask":nd(fb,"attn_mask",mask.numpy())})
    a=ob["velocity"].numpy().astype(np.float64).reshape(-1); b=ref.astype(np.float64).reshape(-1)
    print(f"SPLIT-CHAIN cosine (A->B vs monolithic fp32): {float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)):.6f}", flush=True)
asyncio.run(run())
import shutil; shutil.rmtree(aim.parent, ignore_errors=True)
print("SPLIT SMOKE DONE", flush=True)
