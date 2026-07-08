"""2-layer fp16 export smoke test — validate SDPA-mask patch + coreai_torch fp16
lowering + save/load/forward, before the real 36-layer run."""
import sys, numpy as np, torch
sys.path.insert(0, "/Users/kevinsaltarelli/Dev/Github/coreai-fabric/models/molmoact2")
from action_expert import ActionExpert, AEConfig
from coreai_torch import TorchConverter, get_decomp_table

_orig = torch.nn.functional.scaled_dot_product_attention
def _safe(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
    if attn_mask is None and not is_causal:
        lq, lk = q.shape[-2], k.shape[-2]
        flag = (k.abs().sum() >= -1.0)
        attn_mask = flag.reshape(1, 1).expand(lq, lk)
    return _orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw)
torch.nn.functional.scaled_dot_product_attention = _safe

cfg = AEConfig(num_layers=2)  # tiny
ae = ActionExpert(cfg, llm_kv_dim=1024).eval()
NL, H, ADIM, CTX = 2, 10, 32, 64

class Step(torch.nn.Module):
    def __init__(s): super().__init__(); s.ae = ae
    def forward(s, noisy_actions, timestep, ctx_k, ctx_v):
        kv = [(ctx_k[i], ctx_v[i]) for i in range(NL)]
        return s.ae.forward(noisy_actions, timestep, encoder_kv_states=kv)

step = Step().eval()
g = torch.Generator().manual_seed(0)
o = (torch.randn(1,H,ADIM,generator=g), torch.rand(1,generator=g),
     torch.randn(NL,1,CTX,1024,generator=g), torch.randn(NL,1,CTX,1024,generator=g))
with torch.no_grad(): ref = step(*o).float().numpy()
print("fp32 ref", ref.shape, flush=True)

step16 = step.half().eval()
o16 = tuple(t.half() if t.is_floating_point() else t for t in o)
with torch.no_grad(): ep = torch.export.export(step16, args=o16, strict=False)
ep = ep.run_decompositions(get_decomp_table())
conv = TorchConverter()
conv.add_exported_program(ep, input_names=["noisy_actions","timestep","ctx_k","ctx_v"],
                          output_names=["velocity"], entrypoint_name="action_denoise_step")
prog = conv.to_coreai(); prog.optimize()
from pathlib import Path
aim = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric/build/_smoke/smoke.aimodel")
aim.parent.mkdir(parents=True, exist_ok=True); prog.save_asset(aim)
print("saved+optimized fp16 asset", flush=True)

import asyncio
from coreai.runtime import AIModel, NDArray
async def run():
    mm = await AIModel.load(str(aim)); fn = mm.load_function("action_denoise_step")
    print("LOADS OK", flush=True)
    def nd(n,a):
        dt=np.dtype(str(fn.desc.input_descriptor(n).dtype)); return NDArray(np.asarray(a).astype(dt))
    out = await fn(inputs={"noisy_actions":nd("noisy_actions",o[0].numpy()),"timestep":nd("timestep",o[1].numpy()),
                           "ctx_k":nd("ctx_k",o[2].numpy()),"ctx_v":nd("ctx_v",o[3].numpy())})
    a=out["velocity"].numpy().astype(np.float64).reshape(-1); b=ref.astype(np.float64).reshape(-1)
    cos=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
    print(f"SMOKE cosine (fp16 asset vs fp32 ref): {cos:.6f}", flush=True)
asyncio.run(run())
print("SMOKE DONE", flush=True)
