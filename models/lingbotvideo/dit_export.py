"""LingBot-Video Dense 1.3B — DiT denoise-step export + graph_output_cosine parity.

The LingBotVideoTransformer3DModel is diffusers-based (ModelMixin/ConfigMixin), so the
REAL model loads in the toolchain venv directly (no transformers version conflict) — the
modeling code is vendored-by-path from the upstream repo (reviewed: torch+diffusers only).
Reference, export, lowering and Gate B all run in ONE venv.

Deployable graph = one denoise step: (hidden_states latent [1,16,T,Hl,Wl], timestep [1],
encoder_hidden_states text [1,L,2560]) -> velocity [1,16,T,Hl,Wl]. Host owns the flow-UniPC
sampler loop, the text encoder, and the VAE decode (published). Dense (num_experts 0),
non-causal. fp32. If the monolithic graph exceeds the 0x10004 ceiling, split (this driver
tries whole first and reports)."""
import sys, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric")
SCRATCH = Path("/private/tmp/claude-501/-Users-kevinsaltarelli-Dev-Github-coreai-fabric--claude-worktrees-beautiful-dhawan-6be758/a1dd52aa-599c-41ba-8f12-bac547994f54/scratchpad")
sys.path.insert(0, str(SCRATCH / "lingbot-video"))
OUT = ROOT / "build/lingbot-video-dense-1.3b-dit"
N_OBS, SEED = 8, 0
# fixed static latent + text envelope
C, T, HL, WL, LTXT, TXTDIM = 16, 1, 32, 32, 64, 2560
DRY = "--dry" in sys.argv


def build():
    import glob
    from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel
    tdir = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--robbyant--lingbot-video-dense-1.3b/snapshots/*/transformer"))[0]
    m = LingBotVideoTransformer3DModel.from_pretrained(tdir, torch_dtype=torch.float32).eval()
    print("DiT loaded,", round(sum(p.numel() for p in m.parameters())/1e9, 3), "B params", flush=True)
    return m


def apply_rotary_emb_real(x, rot):
    """Real-arithmetic RoPE (coreai_torch can't lower complex64). Equivalent to the
    upstream complex apply: x viewed as consecutive (x0,x1) pairs, rot = stack([cos,sin])
    shape (2,S,D/2). (x0+i x1)(cos+i sin) -> [x0*cos - x1*sin, x0*sin + x1*cos]."""
    cos = rot[0][None, :, None, :]      # (1,S,1,D/2)
    sin = rot[1][None, :, None, :]
    xr = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = xr[..., 0], xr[..., 1]     # (B,S,H,D/2)
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    out = torch.stack([o0, o1], dim=-1).flatten(3)
    return out.type_as(x)


class Step(torch.nn.Module):
    """B=1 / dense / no-mask / no-parallel denoise step. Replicates the real forward with
    (1) STATIC text length (upstream uses a data-dependent .tolist()), and (2) a BAKED
    real cos/sin RoPE buffer computed once eagerly (upstream's rope is lazy/complex/
    data-dependent). cp_* are nn.Identity. Validated numerically vs the real forward."""
    def __init__(s, m, rot):
        super().__init__(); s.m = m; s.register_buffer("rot", rot)
    def forward(s, hidden_states, timestep, encoder_hidden_states):
        m = s.m
        B, C, T, H, W = hidden_states.shape
        pF, pH, pW = m.config.patch_size
        gt, gh, gw = T // pF, H // pH, W // pW
        n_video = gt * gh * gw
        patch_tokens = hidden_states.reshape(B, C, gt, pF, gh, pH, gw, pW)
        patch_tokens = patch_tokens.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, n_video, pF * pH * pW * C)
        x = m.patch_embedder(patch_tokens)
        text = m.text_embedder(encoder_hidden_states)
        joint = torch.cat([x, text], dim=1)
        S = joint.shape[1]
        rotary = s.rot                     # baked real (2,S,D/2)
        t_emb = m.time_embedder(m.time_proj(timestep.float()))
        temb_input = t_emb.unsqueeze(1).expand(B, S, -1)
        temb6 = m.time_modulation(temb_input.reshape(B * S, -1)).reshape(B, S, -1).reshape(B * S, -1)
        for block in m.blocks:
            joint = block(joint, temb6, rotary, None, None, packed_indices=None, parallel_config=None)
        final_mod = m.norm_out_modulation(temb_input.reshape(joint.shape[0] * joint.shape[1], -1))
        shift, scale = final_mod.reshape(joint.shape[0], joint.shape[1], -1).chunk(2, dim=-1)
        final_hidden = m.norm_out(joint) * (1.0 + scale) + shift
        projected = m.proj_out(final_hidden.to(m.proj_out.weight.dtype))
        x = projected[:, :n_video]
        Cout = m.config.out_channels
        x = x.reshape(B, gt, gh, gw, pF, pH, pW, Cout).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, Cout, T, H, W)
        return x


def obs(i):
    g = torch.Generator().manual_seed(SEED + i)
    hs = torch.randn(1, C, T, HL, WL, generator=g)
    ts = torch.tensor([500.0]) + i  # vary timestep per obs
    txt = torch.randn(1, LTXT, TXTDIM, generator=g)
    return (hs, ts, txt)


def main():
    from coreai_torch import TorchConverter, get_decomp_table
    import lingbot_video.transformer_lingbot_video as tv
    from lingbot_video.transformer_lingbot_video import make_joint_position_ids
    m = build()
    obses = [obs(i) for i in range(N_OBS)]
    # 1) REFERENCE via the ORIGINAL (complex, unpatched) forward
    with torch.no_grad():
        refs = [m(hidden_states=o[0], timestep=o[1], encoder_hidden_states=o[2],
                  return_dict=False)[0].float().numpy() for o in obses]
    # 2) bake the real cos/sin RoPE for the FIXED shape (gt,gh,gw + text len L)
    gt, gh, gw = T // 1, HL // 2, WL // 2
    with torch.no_grad():
        freqs = m.rope(make_joint_position_ids(LTXT, gt, gh, gw, torch.device("cpu")))  # (S, D/2) complex64
    rot = torch.stack([freqs.real.float(), freqs.imag.float()], dim=0).contiguous()      # (2,S,D/2)
    # 3) monkeypatch the complex apply with the real equivalent (export path only)
    tv.apply_rotary_emb = apply_rotary_emb_real
    step = Step(m, rot).eval()
    with torch.no_grad():
        step_out = [step(*o).float().numpy() for o in obses]
    vcos = [float(np.dot(a.reshape(-1), b.reshape(-1)) /
                  (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            for a, b in zip(step_out, refs)]
    print(f"Step(real-RoPE)-vs-real-forward min cosine: {min(vcos):.10f}", flush=True)
    assert min(vcos) > 0.9999, f"exportable Step diverges from real forward: {min(vcos)}"
    print("torch fp32 ref OK, velocity", refs[0].shape, flush=True)
    if DRY:
        print("DRY OK — real-RoPE Step validated vs complex forward", flush=True); return

    with torch.no_grad():
        ep = torch.export.export(step, args=obses[0], strict=False)
    ep = ep.run_decompositions(get_decomp_table())
    print("exported + decomposed", flush=True)
    conv = TorchConverter()
    conv.add_exported_program(ep, input_names=["hidden_states", "timestep", "encoder_hidden_states"],
                              output_names=["velocity"], entrypoint_name="dit_denoise_step")
    prog = conv.to_coreai(); prog.optimize()
    OUT.mkdir(parents=True, exist_ok=True)
    aim = OUT / f"{OUT.name}.aimodel"; prog.save_asset(aim)
    sz = sum(f.stat().st_size for f in aim.rglob("*") if f.is_file())
    print(f"ok: saved fp32 {aim} (~{sz/1e9:.2f} GB)", flush=True)
    import gc; del prog, conv, ep; gc.collect()

    import asyncio
    from coreai.runtime import AIModel, NDArray
    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _environment
    NAMES = ["hidden_states", "timestep", "encoder_hidden_states"]

    async def run():
        mm = await AIModel.load(str(aim)); fn = mm.load_function("dit_denoise_step")
        print("LOADS OK", flush=True)
        def nd(n, a):
            dt = np.dtype(str(fn.desc.input_descriptor(n).dtype)); return NDArray(np.asarray(a).astype(dt))
        cos = []
        for o, r in zip(obses, refs):
            ins = {nm: nd(nm, t.numpy()) for nm, t in zip(NAMES, o)}
            out = await fn(inputs=ins)
            a = out["velocity"].numpy().astype(np.float64).reshape(-1); b = r.astype(np.float64).reshape(-1)
            cos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        cc = np.asarray(cos)
        res = {"metric": "graph_output_cosine", "value": float(cc.min()), "status": "measured",
               "min_cosine": float(cc.min()), "median_cosine": float(np.median(cc)),
               "mean_cosine": float(cc.mean()), "per_obs_cosine": [float(x) for x in cos],
               "n_obs": N_OBS, "reference_dtype": "float32", "quantization": "none", "asset_bytes": int(sz),
               "runner": f"coreai-fabric-parity-runner/{__version__}", "environment": _environment(),
               "reference": "Real LingBotVideoTransformer3DModel (diffusers, fp32) denoise-step velocity vs the .aimodel over seeded (latent, timestep, text) inputs."}
        (OUT / "graph-output-parity-measured.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({k: res[k] for k in ("value", "min_cosine", "median_cosine", "n_obs")}, indent=2))
    asyncio.run(run())
    print("DONE", flush=True)


main()
