"""EO-1 action-denoise-step — int8 combined export + graph_output_cosine parity.
Deployable graph (no-cache): host provides inputs_embeds (VL+text+state prefix +
action-token slots, bypassing the vision tower); the graph runs embed_suffix on the
action tokens -> the Qwen2.5-VL LM forward (use_cache=False) -> action positions ->
action_out_proj -> velocity. Host owns embed_prefix + the Euler loop.
Proven recipe: int8 weight-only + delete-after-refs. (Qwen2.5-VL RoPE is real, no
complex128 issue.)"""
import os, sys, json, shutil
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric")
EO1 = ROOT / "build/_eo1"
OUT = ROOT / "build/eo1-3b"
N_OBS, SEED = 8, 0
PREFIX = 24          # synthetic VL+text prefix length
DRY = "--dry" in sys.argv


def main():
    from transformers import AutoModel
    from coreai_torch import TorchConverter, get_decomp_table
    from torchao.quantization import quantize_, Int8WeightOnlyConfig

    m = AutoModel.from_pretrained(str(EO1), trust_remote_code=True, dtype=torch.float32).eval()
    cfg = m.config
    H = cfg.hidden_size
    CHUNK = cfg.action_chunk_size
    ADIM = cfg.max_action_dim
    SEQ = PREFIX + CHUNK
    print(f"loaded EO-1: hidden={H} chunk={CHUNK} action_dim={ADIM}", flush=True)

    lm = m.vlm_backbone.model  # the Qwen2.5 LM (36 layers)

    class Step(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.m = m; s.lm = lm

        def forward(s, prefix_embeds, timestep, noisy_actions, position_ids):
            # embed the noisy action tokens (action_in_proj + sinusoidal time)
            act_embs = s.m.embed_suffix(timestep, noisy_actions)          # [B, CHUNK, H]
            inputs_embeds = torch.cat([prefix_embeds, act_embs.to(prefix_embeds.dtype)], dim=1)
            out = s.lm(inputs_embeds=inputs_embeds, position_ids=position_ids,
                       use_cache=False, return_dict=True)
            hs = out.last_hidden_state[:, -CHUNK:]                        # action positions
            hs = hs.type(s.m.action_out_proj[0].weight.dtype if hasattr(s.m.action_out_proj,'__getitem__') else torch.float32)
            v = s.m.action_out_proj(hs)
            return v.reshape(noisy_actions.shape)

    step = Step().eval()

    def obs(i):
        g = torch.Generator().manual_seed(SEED + i)
        pe = torch.randn(1, PREFIX, H, generator=g)
        ts = torch.rand(1, generator=g)
        na = torch.randn(1, CHUNK, ADIM, generator=g)
        # Qwen2.5-VL mRoPE position_ids: [3, B, SEQ] (t,h,w) — use a simple linear layout
        pid = torch.arange(SEQ).view(1, 1, SEQ).expand(3, 1, SEQ).contiguous()
        return pe, ts, na, pid

    if DRY:
        obses = [obs(i) for i in range(N_OBS)]
        with torch.no_grad(): refs = [step(*o).float().numpy() for o in obses]
        print("torch ref OK, out", refs[0].shape, flush=True)
        print("DRY OK — build+load+forward validated", flush=True); return

    quantize_(m, Int8WeightOnlyConfig())
    print("int8 quantized", flush=True)
    obses = [obs(i) for i in range(N_OBS)]
    with torch.no_grad(): refs = [step(*o).float().numpy() for o in obses]   # int8 refs
    print("torch ref (int8) OK, out", refs[0].shape, flush=True)

    with torch.no_grad(): ep = torch.export.export(step, args=obses[0], strict=False)
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(ep, input_names=["prefix_embeds","timestep","noisy_actions","position_ids"],
                              output_names=["velocity"], entrypoint_name="action_denoise_step")
    prog = conv.to_coreai(); prog.optimize()
    OUT.mkdir(parents=True, exist_ok=True); aim = OUT / f"{OUT.name}.aimodel"; prog.save_asset(aim)
    sz = sum(f.stat().st_size for f in aim.rglob("*") if f.is_file())
    print(f"ok: saved int8 {aim} (~{sz/1e9:.2f} GB)", flush=True)
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
        for (pe, ts, na, pid), r in zip(obses, refs):
            o = await fn(inputs={"prefix_embeds": nd("prefix_embeds", pe.numpy()), "timestep": nd("timestep", ts.numpy()),
                                 "noisy_actions": nd("noisy_actions", na.numpy()), "position_ids": nd("position_ids", pid.numpy())})
            a = o["velocity"].numpy().astype(np.float64).reshape(-1); b = r.astype(np.float64).reshape(-1)
            cos.append(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)))
        c = np.asarray(cos)
        res = {"metric":"graph_output_cosine","value":float(c.min()),"status":"measured","min_cosine":float(c.min()),
               "median_cosine":float(np.median(c)),"mean_cosine":float(c.mean()),"per_obs_cosine":[float(x) for x in cos],
               "n_obs":N_OBS,"reference_dtype":"float32","quantization":"int8_weight_only","asset_bytes":int(sz),
               "runner":f"coreai-fabric-parity-runner/{__version__}","environment":_environment(),
               "reference":"Torch EO-1 no-cache action-denoise-step velocity vs int8 .aimodel over seeded inputs (host owns VL prefix embeds)."}
        (OUT/"graph-output-parity-measured.json").write_text(json.dumps(res,indent=2)+"\n")
        print(json.dumps({k:res[k] for k in ("value","min_cosine","n_obs")}, indent=2))
    asyncio.run(run())
    print("DONE", flush=True)


main()
