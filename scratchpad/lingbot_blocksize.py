"""Find the max block size (layers per sub-graph) that loads on the ANE.
Random weights — LOAD depends on graph structure, not values. int8-experts + fp16,
mirroring the real export. Usage: python lingbot_blocksize.py <L>"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, "/Users/kevinsaltarelli/Dev/Github/coreai-fabric/models/lingbotvla")
from action_expert import DecoderLayer, LVConfig
from coreai_torch import TorchConverter, get_decomp_table

L = int(sys.argv[1]) if len(sys.argv) > 1 else 6
CHUNK, PREFIX = 32, 32
c = LVConfig(n_action_steps=CHUNK)
KV, HD, SUF = c.n_kv_heads, c.head_dim, CHUNK + 1


class Block(torch.nn.Module):
    def __init__(s, n):
        super().__init__()
        s.c = c
        s.layers = torch.nn.ModuleList([DecoderLayer(c) for _ in range(n)])
        for l in s.layers:
            for p in (l.mlp.experts.gate_proj, l.mlp.experts.up_proj, l.mlp.experts.down_proj):
                torch.nn.init.normal_(p, std=0.02)

    def quant(s):
        for l in s.layers:
            e = l.mlp.experts
            for name in ("gate_proj", "up_proj", "down_proj"):
                w = getattr(e, name).data.float()
                sc = w.abs().amax(dim=2, keepdim=True) / 127.0
                q = torch.round(w / sc.clamp(min=1e-12)).clamp(-127, 127).to(torch.int8)
                delattr(e, name)
                e.register_buffer(name + "_q", q)
                e.register_buffer(name + "_s", sc.half())

    def forward(s, h, cond, cos, sin, prefix_k, prefix_v, attn_mask):
        for i, l in enumerate(s.layers):
            h = l(h, cond, prefix_k[i], prefix_v[i], cos, sin, attn_mask)
        return h


b = Block(L).eval()
if "--int8" in sys.argv: b.quant()
b = b.half().eval()
g = torch.Generator().manual_seed(0)
o = (torch.randn(1, SUF, c.hidden, generator=g).half(), torch.randn(1, c.hidden, generator=g).half(),
     torch.randn(1, SUF, HD, generator=g).half(), torch.randn(1, SUF, HD, generator=g).half(),
     torch.randn(L, 1, PREFIX, KV, HD, generator=g).half(), torch.randn(L, 1, PREFIX, KV, HD, generator=g).half(),
     torch.ones(1, SUF, PREFIX + SUF, dtype=torch.bool))
with torch.no_grad():
    ep = torch.export.export(b, args=o, strict=False)
ep = ep.run_decompositions(get_decomp_table())
conv = TorchConverter()
conv.add_exported_program(ep, input_names=["h", "cond", "cos", "sin", "prefix_k", "prefix_v", "attn_mask"],
                          output_names=["h_out"], entrypoint_name="block")
prog = conv.to_coreai(); prog.optimize()
aim = Path(f"/Users/kevinsaltarelli/Dev/Github/coreai-fabric/build/_bsz/b{L}.aimodel")
aim.parent.mkdir(parents=True, exist_ok=True); prog.save_asset(aim)
sz = sum(f.stat().st_size for f in aim.rglob("*") if f.is_file())
print(f"L={L}: saved {sz/1e9:.2f}GB", flush=True)
import asyncio
from coreai.runtime import AIModel
async def run():
    m = await AIModel.load(str(aim)); m.load_function("block")
    print(f"L={L}: LOADS OK", flush=True)
try:
    asyncio.run(run())
    print(f"L={L}: RESULT loadable", flush=True)
except Exception as e:
    print(f"L={L}: RESULT FAIL {str(e)[:80]}", flush=True)
import shutil; shutil.rmtree(aim.parent, ignore_errors=True)
