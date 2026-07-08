"""2-layer int8 export smoke — validate coreai_torch lowering of the MoE ops
(topk/one_hot/gather/3D-einsum/where) + eager attn + AdaRMSNorm + rope, before the
full 36-layer run."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "/Users/kevinsaltarelli/Dev/Github/coreai-fabric/models/lingbotvla")
from action_expert import LingbotActionExpert, LVConfig
from coreai_torch import TorchConverter, get_decomp_table
from torchao.quantization import quantize_, Int8WeightOnlyConfig

CHUNK, PREFIX = 8, 8
c = LVConfig(num_layers=2, n_action_steps=CHUNK)
m = LingbotActionExpert(c).eval()
# init fused expert params (empty -> fill)
for l in m.layers:
    for p in (l.mlp.experts.gate_proj, l.mlp.experts.up_proj, l.mlp.experts.down_proj):
        torch.nn.init.normal_(p, std=0.02)
NL, KV, HD = c.num_layers, c.n_kv_heads, c.head_dim
SUF = CHUNK + 1

class Step(torch.nn.Module):
    def __init__(s): super().__init__(); s.m = m
    def forward(s, state, na, ts, pk, pv, pos, mask):
        return s.m.forward(state, na, ts, pk, pv, pos, mask)
step = Step().eval()
g = torch.Generator().manual_seed(0)
o = (torch.randn(1,c.state_dim,generator=g), torch.randn(1,CHUNK,c.action_dim,generator=g), torch.rand(1,generator=g),
     torch.randn(NL,1,PREFIX,KV,HD,generator=g), torch.randn(NL,1,PREFIX,KV,HD,generator=g),
     torch.arange(PREFIX,PREFIX+SUF).view(1,SUF), torch.ones(1,SUF,PREFIX+SUF,dtype=torch.bool))
with torch.no_grad(): ref = step(*o).float().numpy()
print("fp32 ref", ref.shape, flush=True)
step = step.half().eval()
o = tuple(t.half() if t.is_floating_point() else t for t in o)
with torch.no_grad(): ep = torch.export.export(step, args=o, strict=False)
ep = ep.run_decompositions(get_decomp_table())
conv = TorchConverter()
conv.add_exported_program(ep, input_names=["state","noisy_actions","timestep","prefix_k","prefix_v","position_ids","attn_mask"],
                          output_names=["velocity"], entrypoint_name="action_denoise_step")
prog = conv.to_coreai(); prog.optimize()
aim = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric/build/_lsmoke/s.aimodel")
aim.parent.mkdir(parents=True, exist_ok=True); prog.save_asset(aim)
print("saved+optimized int8", flush=True)
import asyncio
from coreai.runtime import AIModel, NDArray
NAMES=["state","noisy_actions","timestep","prefix_k","prefix_v","position_ids","attn_mask"]
async def run():
    mm=await AIModel.load(str(aim)); fn=mm.load_function("action_denoise_step")
    print("LOADS OK", flush=True)
    def nd(n,a):
        dt=np.dtype(str(fn.desc.input_descriptor(n).dtype)); return NDArray(np.asarray(a).astype(dt))
    out=await fn(inputs={nm:nd(nm,t.numpy()) for nm,t in zip(NAMES,o)})
    a=out["velocity"].numpy().astype(np.float64).reshape(-1); b=ref.astype(np.float64).reshape(-1)
    print(f"SMOKE cosine {float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)):.6f}", flush=True)
asyncio.run(run())
print("SMOKE DONE", flush=True)
