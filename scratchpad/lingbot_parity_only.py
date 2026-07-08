"""Load the already-exported lingbot-vla-v2 .aimodel + run graph_output_cosine parity
(recompute fp32 refs; skip re-export)."""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "/Users/kevinsaltarelli/Dev/Github/coreai-fabric/models/lingbotvla")
from action_expert import LingbotActionExpert, LVConfig
ROOT = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric")
WEIGHTS = ROOT / "build/_lingbotvla/qwen_expert.safetensors"
OUT = ROOT / "build/lingbot-vla-v2"; aim = OUT / "lingbot-vla-v2.aimodel"
N_OBS, SEED, CHUNK, PREFIX = 8, 0, 32, 32
c = LVConfig(n_action_steps=CHUNK); m = LingbotActionExpert(c).eval()
from safetensors.torch import load_file
raw = load_file(str(WEIGHTS)); PRE = "qwenvl_with_expert.qwen_expert.model."
sd = {}
for k, v in raw.items():
    if k.startswith(PRE): sd[k[len(PRE):]] = v
    elif k.startswith(("action_in_proj","action_out_proj","action_time_mlp","state_proj")): sd[k] = v
m.load_state_dict(sd, strict=False)
NL, KV, HD, SUF = c.num_layers, c.n_kv_heads, c.head_dim, CHUNK + 1
def obs(i):
    g = torch.Generator().manual_seed(SEED + i)
    return (torch.randn(1,c.state_dim,generator=g), torch.randn(1,CHUNK,c.action_dim,generator=g), torch.rand(1,generator=g),
            torch.randn(NL,1,PREFIX,KV,HD,generator=g), torch.randn(NL,1,PREFIX,KV,HD,generator=g),
            torch.arange(PREFIX,PREFIX+SUF).view(1,SUF), torch.ones(1,SUF,PREFIX+SUF,dtype=torch.bool))
obses = [obs(i) for i in range(N_OBS)]
with torch.no_grad(): refs = [m.forward(*o).float().numpy() for o in obses]
print("fp32 refs OK", refs[0].shape, flush=True)
import asyncio
from coreai.runtime import AIModel, NDArray
from coreai_fabric import __version__
from coreai_fabric.parity_runner import _environment
NAMES = ["state","noisy_actions","timestep","prefix_k","prefix_v","position_ids","attn_mask"]
async def run():
    mm = await AIModel.load(str(aim)); fn = mm.load_function("action_denoise_step")
    print("LOADS OK", flush=True)
    def nd(n,a):
        dt=np.dtype(str(fn.desc.input_descriptor(n).dtype)); return NDArray(np.asarray(a).astype(dt))
    cos=[]
    for o,r in zip(obses,refs):
        out=await fn(inputs={nm:nd(nm,t.numpy()) for nm,t in zip(NAMES,o)})
        a=out["velocity"].numpy().astype(np.float64).reshape(-1); b=r.astype(np.float64).reshape(-1)
        cos.append(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)))
    cc=np.asarray(cos); sz=sum(f.stat().st_size for f in aim.rglob("*") if f.is_file())
    res={"metric":"graph_output_cosine","value":float(cc.min()),"status":"measured","min_cosine":float(cc.min()),
         "median_cosine":float(np.median(cc)),"mean_cosine":float(cc.mean()),"per_obs_cosine":[float(x) for x in cos],
         "n_obs":N_OBS,"reference_dtype":"float32","quantization":"float16","asset_bytes":int(sz),
         "runner":f"coreai-fabric-parity-runner/{__version__}","environment":_environment(),
         "reference":"Torch LingBot-VLA-2.0 MoE action-expert denoise-step velocity vs fp16 .aimodel over seeded inputs (host owns Qwen3-VL prefix K/V + Euler loop). MoE dense-fused (exact vs sparse top-4)."}
    (OUT/"graph-output-parity-measured.json").write_text(json.dumps(res,indent=2)+"\n")
    print(json.dumps({k:res[k] for k in ("value","min_cosine","median_cosine","n_obs")},indent=2))
asyncio.run(run())
print("DONE", flush=True)
