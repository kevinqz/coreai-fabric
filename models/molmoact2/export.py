"""MolmoAct2-LIBERO action-denoise-step — fp16 export + graph_output_cosine parity.

Deployable graph = the SEPARABLE action expert (36 blocks, hidden 768) doing one
flow-matching denoise step. It cross-attends, per block, to the VLM's per-layer K/V
(fed as the `ctx_k`/`ctx_v` graph inputs — the host prefills the Molmo2 VLM once with
`collect_layer_kv_states=True`). The host owns the VLM, the Euler loop
(trajectory += dt*velocity, num_flow_timesteps=8) and un-normalization. This mirrors
the FastWAM lane (video-KV-cache as inputs) but for a VLM backbone.

The action expert is ~575M params → fp16 ~1.15GB, which loads on the ANE with no need
for int8 (unlike the whole-VLM lanes EO-1/FastWAM). RoPE is real cos/sin already
(no complex-dtype rewrite). SDPA runs mask-free (causal_attn=False, all context
valid); we inject a DATA-DEPENDENT all-true bool keep-mask so the MPSGraph
FoldMultiplyIntoSDPAScale path (mask-free SDPA segfault on macOS 27) never triggers."""
import os, sys, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric")
sys.path.insert(0, str(ROOT / "models/molmoact2"))
WEIGHTS = ROOT / "build/_molmoact2/action_expert.safetensors"
OUT = ROOT / "build/molmoact2-libero"
N_OBS, SEED = 8, 0
CHUNK = 10            # LIBERO denoise horizon (config chunk_size = n_action_steps = 10)
CTX_SEQ = 64          # synthetic VLM context length for parity (host prefills the real length)
LLM_KV_DIM = 1024     # VLM num_key_value_heads(8) * head_dim(128)
DRY = "--dry" in sys.argv


def _patch_sdpa():
    """Inject a data-dependent all-true bool keep-mask when SDPA is called mask-free,
    so the mask-free FoldMultiplyIntoSDPAScale segfault (macOS 27) cannot trigger."""
    _orig = torch.nn.functional.scaled_dot_product_attention

    def _safe(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
        if attn_mask is None and not is_causal:
            lq, lk = q.shape[-2], k.shape[-2]
            flag = (k.abs().sum() >= -1.0)          # scalar bool, always True, data-dependent
            attn_mask = flag.reshape(1, 1).expand(lq, lk)
        return _orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw)

    torch.nn.functional.scaled_dot_product_attention = _safe


def main():
    from action_expert import ActionExpert, AEConfig
    from coreai_torch import TorchConverter, get_decomp_table
    from safetensors.torch import load_file

    _patch_sdpa()
    cfg = AEConfig()  # hidden 768, 36 layers, 8 heads, max_action_dim 32, horizon 32
    ae = ActionExpert(cfg, llm_kv_dim=LLM_KV_DIM).eval()
    sd = load_file(str(WEIGHTS))
    missing, unexpected = ae.load_state_dict(sd, strict=False)
    print(f"loaded {len(sd)} tensors; missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    if missing:
        print("  MISSING sample:", missing[:8], flush=True)
    if unexpected:
        print("  UNEXPECTED sample:", unexpected[:8], flush=True)
    NL = cfg.num_layers
    H, ADIM = CHUNK, cfg.max_action_dim

    class Step(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.ae = ae

        def forward(s, noisy_actions, timestep, ctx_k, ctx_v):
            kv = [(ctx_k[i], ctx_v[i]) for i in range(NL)]
            return s.ae.forward(noisy_actions, timestep, encoder_kv_states=kv)

    step = Step().eval()

    def obs(i):
        g = torch.Generator().manual_seed(SEED + i)
        na = torch.randn(1, H, ADIM, generator=g)
        ts = torch.rand(1, generator=g)
        ck = torch.randn(NL, 1, CTX_SEQ, LLM_KV_DIM, generator=g)
        cv = torch.randn(NL, 1, CTX_SEQ, LLM_KV_DIM, generator=g)
        return na, ts, ck, cv

    obses = [obs(i) for i in range(N_OBS)]
    with torch.no_grad():
        refs = [step(*o).float().numpy() for o in obses]
    print("torch ref OK, out", refs[0].shape, flush=True)
    if DRY:
        print("DRY OK — build+load+forward validated", flush=True); return

    # fp16 weights (small model -> loads on ANE without int8; fp32 refs above)
    step_fp16 = step.half().eval()
    with torch.no_grad():
        ep = torch.export.export(step_fp16, args=tuple(t.half() if t.is_floating_point() else t for t in obses[0]), strict=False)
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(ep, input_names=["noisy_actions", "timestep", "ctx_k", "ctx_v"],
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

    async def run():
        mm = await AIModel.load(str(aim)); fn = mm.load_function("action_denoise_step")
        print("LOADS OK", flush=True)

        def nd(n, a):
            dt = np.dtype(str(fn.desc.input_descriptor(n).dtype)); return NDArray(np.asarray(a).astype(dt))
        cos = []
        for (na, ts, ck, cv), r in zip(obses, refs):
            o = await fn(inputs={"noisy_actions": nd("noisy_actions", na.numpy()), "timestep": nd("timestep", ts.numpy()),
                                 "ctx_k": nd("ctx_k", ck.numpy()), "ctx_v": nd("ctx_v", cv.numpy())})
            a = o["velocity"].numpy().astype(np.float64).reshape(-1); b = r.astype(np.float64).reshape(-1)
            cos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        c = np.asarray(cos)
        res = {"metric": "graph_output_cosine", "value": float(c.min()), "status": "measured", "min_cosine": float(c.min()),
               "median_cosine": float(np.median(c)), "mean_cosine": float(c.mean()), "per_obs_cosine": [float(x) for x in cos],
               "n_obs": N_OBS, "reference_dtype": "float32", "quantization": "float16", "asset_bytes": int(sz),
               "runner": f"coreai-fabric-parity-runner/{__version__}", "environment": _environment(),
               "reference": "Torch MolmoAct2 action-expert action-denoise-step velocity vs fp16 .aimodel over seeded inputs (host owns VLM per-layer K/V prefill)."}
        (OUT / "graph-output-parity-measured.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({k: res[k] for k in ("value", "min_cosine", "n_obs")}, indent=2))
    asyncio.run(run())
    print("DONE", flush=True)


main()
