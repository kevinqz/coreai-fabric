"""FastWAM action-denoise-step — int8 combined export + graph_output_cosine parity.
The deployable graph is the ACTION path only (action_expert + MoT action-cached);
the video expert is never forwarded (its per-layer K/V arrive as `video_kv_cache`
graph inputs), so we build a tiny video STUB just to satisfy MoT.__init__.
Proven recipe: real-RoPE monkeypatch + int8 weight-only + delete-after-load."""
import os, sys, json, shutil
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric")
SRC = ROOT / "build/_src_fastwam"
WEIGHTS = ROOT / "build/_fastwam_libero/model.safetensors"
OUT = ROOT / "build/fastwam-libero"
N_OBS, SEED = 8, 0

# action_dit_config (lerobot/fastwam_base)
ADC = dict(action_dim=7, hidden_dim=1024, ffn_dim=4096, num_heads=24, attn_head_dim=128,
           num_layers=30, text_dim=4096, freq_dim=256, eps=1e-6)
VIDEO_SEQ = 32     # synthetic video-cache length (host supplies the real prefill)
ACTION_H = 32      # action horizon
TEXT_LEN = 32


def _stub_lerobot():
    import types
    for n in ("lerobot", "lerobot.utils"):
        sys.modules.setdefault(n, types.ModuleType(n))
    iu = types.ModuleType("lerobot.utils.import_utils")
    iu._diffusers_available = True; iu._transformers_available = True
    iu.require_package = lambda *a, **k: None
    sys.modules["lerobot.utils.import_utils"] = iu
    c = types.ModuleType("lerobot.utils.constants"); c.OBS_STATE = "observation.state"; c.ACTION = "action"
    sys.modules["lerobot.utils.constants"] = c
    pt = types.ModuleType("lerobot.policies.pretrained")
    pt.PreTrainedPolicy = object
    sys.modules["lerobot.policies"] = types.ModuleType("lerobot.policies")
    sys.modules["lerobot.policies.pretrained"] = pt


def _real_rope(vdit):
    """Rewrite the complex apply_dense_rope / rope_apply to real cos/sin."""
    def apply_dense_rope_real(x, freqs, num_heads):
        from einops import rearrange
        x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
        # freqs is now REAL [s, 1, head_dim] (interleaved cos/sin from the real buffer)
        fr = freqs.reshape(*freqs.shape[:-1], -1, 2)
        cos = fr[..., 0].to(torch.float32); sin = fr[..., 1].to(torch.float32)
        xp = x.to(torch.float32).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
        xe, xo = xp[..., 0], xp[..., 1]
        oe = xe * cos - xo * sin; oo = xe * sin + xo * cos
        out = torch.stack([oe, oo], dim=-1).flatten(3)
        return out.to(x.dtype).flatten(2)
    def rope_apply_real(x, freqs, num_heads=None):
        # x already [b, s, n, d]; freqs complex
        cos = freqs.real.to(torch.float32); sin = freqs.imag.to(torch.float32)
        xp = x.to(torch.float32).reshape(*x.shape[:-1], -1, 2)
        xe, xo = xp[..., 0], xp[..., 1]
        oe = xe * cos - xo * sin; oo = xe * sin + xo * cos
        return torch.stack([oe, oo], dim=-1).flatten(-2).to(x.dtype)
    if hasattr(vdit, "apply_dense_rope"): vdit.apply_dense_rope = apply_dense_rope_real
    if hasattr(vdit, "rope_apply"): vdit.rope_apply = rope_apply_real


def main():
    from coreai_torch import TorchConverter, get_decomp_table
    from torchao.quantization import quantize_, Int8WeightOnlyConfig
    sys.path.insert(0, str(SRC)); _stub_lerobot()
    from wan import modular as M
    from wan import video_dit as VD
    _real_rope(VD); _real_rope(M)

    action_expert = M.ActionDiT(**ADC).eval()
    video_stub = M.ActionDiT(**{**ADC, "hidden_dim": 128, "ffn_dim": 256}).eval()  # never forwarded
    mot = M.MoT(mixtures={"video": video_stub, "action": action_expert}).eval()
    # make the complex RoPE buffer REAL (coreai_torch has no complex dtype) — [., d/2, 2]
    for ex in (action_expert, video_stub):
        f = ex.freqs
        ex.freqs = torch.stack([f.real, f.imag], dim=-1).contiguous().to(torch.float32)

    # load only the action-expert weights
    from safetensors import safe_open
    subset = {}
    with safe_open(str(WEIGHTS), "pt", "cpu") as h:
        for k in h.keys():
            for pre in ("model.mot.mixtures.action.", "mot.mixtures.action."):
                if k.startswith(pre):
                    subset[k[len(pre):]] = h.get_tensor(k); break
    missing, unexpected = action_expert.load_state_dict(subset, strict=False)
    print(f"action_expert loaded: {len(subset)} tensors, missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    if len(subset) == 0:
        with safe_open(str(WEIGHTS), "pt", "cpu") as h:
            print("SAMPLE KEYS:", [k for k in list(h.keys()) if "action" in k][:8]); return

    DRY = "--dry" in sys.argv

    class Step(torch.nn.Module):
        def __init__(s): super().__init__(); s.ae = action_expert; s.mot = mot
        def forward(s, latents_action, timestep, context, vk, vv):
            pre = s.ae.pre_dit(latents_action, timestep, context)
            vkv = [{"k": vk[i], "v": vv[i]} for i in range(len(s.mot.layers))]
            total = VIDEO_SEQ + latents_action.shape[1]
            amask = torch.ones(total, total, dtype=torch.bool)
            tok = s.mot.forward_action_with_video_cache(
                action_tokens=pre["tokens"], action_freqs=pre["freqs"], action_t_mod=pre["t_mod"],
                action_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
                video_kv_cache=vkv, attention_mask=amask, video_seq_len=VIDEO_SEQ)
            return s.ae.post_dit(tok, pre)

    step = Step().eval()
    NL = len(mot.layers); NH, HD = ADC["num_heads"], ADC["attn_head_dim"]

    def obs(i):
        g = torch.Generator().manual_seed(SEED + i)
        la = torch.randn(1, ACTION_H, ADC["action_dim"], generator=g)
        ts = torch.randint(0, 1000, (1,), generator=g, dtype=torch.long)
        ctx = torch.randn(1, TEXT_LEN, ADC["text_dim"], generator=g)
        vk = torch.randn(NL, 1, VIDEO_SEQ, NH * HD, generator=g)  # k is 3D [B, seq, inner]
        vv = torch.randn(NL, 1, VIDEO_SEQ, NH * HD, generator=g)
        return la, ts, ctx, vk, vv

    quantize_(action_expert, Int8WeightOnlyConfig())
    print("int8 quantized", flush=True)
    obses = [obs(i) for i in range(N_OBS)]
    with torch.no_grad(): refs = [step(*o).float().numpy() for o in obses]  # int8 refs -> export fidelity
    print("torch ref (int8) OK, out", refs[0].shape, flush=True)
    if DRY:
        print("DRY OK — build+load+int8+forward validated", flush=True); return
    # export is proven; disk is tight -> free the 12GB source now (refs already in RAM).
    if WEIGHTS.exists(): os.remove(WEIGHTS)
    shutil.rmtree(WEIGHTS.parent / ".cache", ignore_errors=True)
    print("freed source weights (refs in RAM)", flush=True)

    with torch.no_grad(): ep = torch.export.export(step, args=obses[0], strict=False)
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(ep, input_names=["latents_action","timestep","context","vk","vv"],
                              output_names=["velocity"], entrypoint_name="action_denoise_step")
    prog = conv.to_coreai(); prog.optimize()
    OUT.mkdir(parents=True, exist_ok=True); aim = OUT / f"{OUT.name}.aimodel"; prog.save_asset(aim)
    sz = sum(f.stat().st_size for f in aim.rglob("*") if f.is_file())
    print(f"ok: saved int8 {aim} (~{sz/1e9:.2f} GB)", flush=True)

    # release coremltools export scratch/mmap before the ANE-compile load (disk is tight)
    import gc
    del prog, conv, ep
    gc.collect()

    import asyncio
    from coreai.runtime import AIModel, NDArray
    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _environment
    async def run():
        m = await AIModel.load(str(aim)); fn = m.load_function("action_denoise_step")
        print("LOADS OK", flush=True)
        def nd(n, a):
            dt = np.dtype(str(fn.desc.input_descriptor(n).dtype)); return NDArray(np.asarray(a).astype(dt))
        cos = []
        for (la, ts, ctx, vk, vv), r in zip(obses, refs):
            o = await fn(inputs={"latents_action": nd("latents_action", la.numpy()), "timestep": nd("timestep", ts.numpy()),
                                 "context": nd("context", ctx.numpy()), "vk": nd("vk", vk.numpy()), "vv": nd("vv", vv.numpy())})
            a = o["velocity"].numpy().astype(np.float64).reshape(-1); b = r.astype(np.float64).reshape(-1)
            cos.append(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)))
        c = np.asarray(cos)
        res = {"metric":"graph_output_cosine","value":float(c.min()),"status":"measured","min_cosine":float(c.min()),
               "median_cosine":float(np.median(c)),"mean_cosine":float(c.mean()),"per_obs_cosine":[float(x) for x in cos],
               "n_obs":N_OBS,"reference_dtype":"float32","quantization":"int8_weight_only","asset_bytes":int(sz),
               "runner":f"coreai-fabric-parity-runner/{__version__}","environment":_environment(),
               "reference":"Torch FastWAM cache-free... action-denoise-step velocity vs int8 .aimodel, seeded inputs."}
        (OUT/"graph-output-parity-measured.json").write_text(json.dumps(res,indent=2)+"\n")
        print(json.dumps({k:res[k] for k in ("value","min_cosine","n_obs")}, indent=2))
    asyncio.run(run())
    print("DONE", flush=True)

main()
