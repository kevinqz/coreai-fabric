"""SmolVLA flow-matching split-export — adapted from models/pi0/export.py.

SmolVLA = a SmolVLM2-500M VLM + a Gemma-style action expert, same flow-matching Euler loop as
pi0/pi05: encode (VLM prefix -> KV cache, run ONCE) + denoise_step (expert, host drives it 10x).
Differences vs pi0: the VLM is SmolVLM2-500M (16 layers, 5 KV heads, head-dim 64); the prefix is
241 tokens (3 cams x 64 SigLIP-shuffle embeds + 48 lang + 1 state); the KV cache is a per-layer
DICT {i: {"key_states","value_states"}} of 32 bf16 tensors shaped (1, 241, 5, 64) — note the seq
axis is dim 1, unlike pi0's (B, heads, seq, head_dim). embed_prefix takes `state` as a separate
input (unlike pi05, which folds it into tokens). Deps: `num2words` + an ONLINE base-VLM fetch.

  export  (venv-A, .venv-lerobot): torch.export -> encode.pt2 / denoise_step.pt2
  --lower (venv-B, .venv + coreai_torch): load the .pt2 -> ONE .aimodel (encode + denoise_step)
"""
import argparse
import contextlib
import os
from pathlib import Path

# Probed on edge-inference/smolvla-so101-pick-orange (SmolVLM2-500M-Video-Instruct).
CHUNK = int(os.environ.get("SMOLVLA_CHUNK", "50"))
ACT_DIM, STATE_DIM = 32, 32
TOK = int(os.environ.get("SMOLVLA_TOK", "48"))                 # tokenizer_max_length
N_LAYERS, KV_HEADS, HEAD_DIM = 16, 5, 64                       # SmolVLM2-500M expert cache dims
PREFIX_LEN = int(os.environ.get("SMOLVLA_PREFIX_LEN", "241"))  # 3*64 img + 48 lang + 1 state
IMG = int(os.environ.get("SMOLVLA_IMG", "512"))                # SmolVLM2 input resolution


def _safe_make_att_2d_masks(pad_masks, att_masks):
    """bool-mul-free 2-D attention mask (the coreai runtime has no bool `mul` kernel) — int32
    mul + compare, identical result. Monkeypatched module-wide (encode + denoise both call it)."""
    import torch
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = (cumsum[:, None, :] <= cumsum[:, :, None]).to(torch.int32)
    pad = pad_masks.to(torch.int32)
    pad_2d = pad[:, None, :] * pad[:, :, None]
    return (att_2d * pad_2d) > 0


@contextlib.contextmanager
def no_deepcopy():
    """denoise_step may deepcopy the KV cache (fails on FakeTensor during torch.export). Identity
    copy is safe: we export a SINGLE step driven with fresh inputs, never mutating a shared cache."""
    import copy as _copy
    orig = _copy.deepcopy
    _copy.deepcopy = lambda x, memo=None: x
    try:
        yield
    finally:
        _copy.deepcopy = orig


def _load_smolvla_config(cfg_dir: str):
    """Allowlist the config to SmolVLAConfig's declared fields (strict draccus decode)."""
    import json
    import tempfile
    import dataclasses
    from pathlib import Path as _P
    import lerobot.policies.smolvla.modeling_smolvla  # noqa: F401 — registers 'smolvla'
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAConfig
    from lerobot.configs.policies import PreTrainedConfig
    raw = json.loads((_P(cfg_dir) / "config.json").read_text())
    valid = {f.name for f in dataclasses.fields(SmolVLAConfig)} | {"type"}
    stripped = {k: v for k, v in raw.items() if k in valid}
    tmp = _P(tempfile.mkdtemp(prefix="smolvlacfg_"))
    (tmp / "config.json").write_text(json.dumps(stripped))
    return PreTrainedConfig.from_pretrained(str(tmp), local_files_only=True)


def _load_policy(fp16: bool = False):
    """venv-A. Loads from SMOLVLA_CONFIG_DIR (a local mirror). Needs ONLINE (base SmolVLM2-500M
    config/processor fetch) + num2words. Pins to CPU (config device=cuda->mps leaks the VLM)."""
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    cfg_dir = os.environ.get("SMOLVLA_CONFIG_DIR", "build/_hf_mirror/smolvla")
    cfg = _load_smolvla_config(cfg_dir)
    policy = SmolVLAPolicy.from_pretrained(cfg_dir, config=cfg, local_files_only=False).eval()
    policy = policy.to("cpu")
    if fp16:
        policy = policy.half()

        def _cast_in(module, args):
            if args and hasattr(args[0], "to"):
                return (args[0].to(module.weight.dtype), *args[1:])
            return args
        m = policy.model
        for attr in ("action_out_proj", "action_in_proj", "state_proj"):
            proj = getattr(m, attr, None)
            if proj is not None and hasattr(proj, "weight"):
                proj.register_forward_pre_hook(_cast_in)
    return policy


def _flatten_cache(pkv):
    """SmolVLA cache dict {i: {key_states, value_states}} -> flat tuple of 2*N_LAYERS tensors."""
    out = []
    for i in sorted(pkv.keys()):
        out.extend([pkv[i]["key_states"], pkv[i]["value_states"]])
    return tuple(out)


def _unflatten_cache(tensors):
    return {i: {"key_states": tensors[2 * i], "value_states": tensors[2 * i + 1]}
            for i in range(N_LAYERS)}


_CACHE_NAMES = [f"{kind}_{i}" for i in range(N_LAYERS) for kind in ("k", "v")]
_ENC_INPUTS = ["img0", "img1", "img2", "imask0", "imask1", "imask2", "lang_tokens", "lang_masks", "state"]
# NOTE: "prefix_embeds" (pe) is emitted as an ANCHOR output. Without it, coreai's lowering
# miscompiles the embed_prefix intermediate that feeds the attention — the cache stays plausible
# for small/synthetic activations but diverges catastrophically for real (large-activation) images
# (real-image action cosine 0.02 vs 0.99 with the anchor). Materializing pe as a graph output
# forces the correct layout. The host ignores this output; denoise_step never consumes it.
_ENC_OUTPUTS = ["prefix_pad_masks", "prefix_embeds", *_CACHE_NAMES]
_DEN_INPUTS = ["prefix_pad_masks", "x_t", "timestep", *_CACHE_NAMES]
_DEN_OUTPUTS = ["velocity"]


def _patched_vision_embed_forward(self, pixel_values, patch_attention_mask=None):
    """coreai_torch has no aten.bucketize/empty_permuted lowering. SmolVLM's vision-embedding
    computes per-patch position ids via bucketize(fractional_coords, boundaries) — but for a
    FIXED full-resolution image (all patches valid) that reduces to identity: pos_ids = arange
    (num_patches), row-major. Replace the bucketize + torch.full path with a plain arange (valid
    for the fixed export shape). Monkeypatched module-wide before export."""
    import torch
    patch_embeds = self.patch_embedding(pixel_values)
    embeddings = patch_embeds.flatten(2).transpose(1, 2)
    bsize = pixel_values.shape[0]
    pos_ids = torch.arange(self.num_patches, device=embeddings.device).unsqueeze(0).expand(bsize, -1)
    return embeddings + self.position_embedding(pos_ids)


def _build_wrappers(fp16: bool = False):
    """venv-A only. Returns (encode_module, denoise_module, dummy_args_dict)."""
    import torch
    from lerobot.policies.smolvla import modeling_smolvla as _ms
    _ms.make_att_2d_masks = _safe_make_att_2d_masks
    from transformers.models.smolvlm import modeling_smolvlm as _sv
    _sv.SmolVLMVisionEmbeddings.forward = _patched_vision_embed_forward

    policy = _load_policy(fp16=fp16)
    m = policy.model

    class EncodeWrapper(torch.nn.Module):
        def forward(self, img0, img1, img2, imask0, imask1, imask2, lang_tokens, lang_masks, state):
            pe, ppad, patt = m.embed_prefix([img0, img1, img2], [imask0, imask1, imask2],
                                            lang_tokens, lang_masks, state=state)
            att2d = _safe_make_att_2d_masks(ppad, patt)
            pos = torch.cumsum(ppad, 1) - 1
            _, pkv = m.vlm_with_expert.forward(
                attention_mask=att2d, position_ids=pos, past_key_values=None,
                inputs_embeds=[pe, None], use_cache=True, fill_kv_cache=True)
            # SmolVLM2's KV cache is bf16 internally; cast to the export float dtype so encode's
            # cache OUTPUTS match denoise's cache INPUTS (coreai rejects a bf16 output where the
            # graph declares fp32/fp16). bf16->fp32 is lossless; ->fp16 matches the fp16 asset.
            flat = tuple(t.to(img0.dtype) for t in _flatten_cache(pkv))
            return (ppad, pe.to(img0.dtype), *flat)  # pe = anchor (see _ENC_OUTPUTS note)

    class DenoiseWrapper(torch.nn.Module):
        def forward(self, prefix_pad_masks, x_t, timestep, *cache_tensors):
            pkv = _unflatten_cache(cache_tensors)
            return m.denoise_step(prefix_pad_masks=prefix_pad_masks, past_key_values=pkv,
                                  x_t=x_t, timestep=timestep)

    import torch as _t
    fd = _t.float16 if fp16 else _t.float32
    d = dict(
        img=_t.zeros(1, 3, IMG, IMG, dtype=fd), imask=_t.ones(1, dtype=_t.bool),
        tok=_t.zeros(1, TOK, dtype=_t.long), lmask=_t.ones(1, TOK, dtype=_t.bool),
        state=_t.zeros(1, STATE_DIM, dtype=fd), ppad=_t.ones(1, PREFIX_LEN, dtype=_t.bool),
        xt=_t.zeros(1, CHUNK, ACT_DIM, dtype=fd), t=_t.zeros(1, dtype=fd),
        cache=tuple(_t.zeros(1, PREFIX_LEN, KV_HEADS, HEAD_DIM, dtype=fd) for _ in range(2 * N_LAYERS)),
    )
    return EncodeWrapper().eval(), DenoiseWrapper().eval(), d


def _disk_free_gb() -> float:
    import shutil
    return shutil.disk_usage("/").free / 1e9


def natural_image(g):
    """One [1,3,IMG,IMG] high-entropy, 1/f-ish image in [-1,1] (sum of many random-phase sinusoids
    across a broad frequency band → detail at all scales). Successive calls (advancing generator g)
    yield distinct images. Used both as a NON-DEGENERATE tracing example and as realistic parity
    input: white-noise/zeros are OOD for the vision tower, which BOTH mis-specializes the exported
    graph (catastrophic real-image cache) AND ill-conditions the parity cosine."""
    import torch
    lin = torch.linspace(-1, 1, IMG)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")

    def chan():
        acc = torch.zeros(IMG, IMG)
        for _ in range(40):
            fx, fy = (torch.rand(2, generator=g) * 16 + 0.5).tolist()
            ph = torch.rand(1, generator=g).item() * 6.283
            amp = 1.0 / (abs(fx) + abs(fy) + 1.0)
            acc = acc + amp * torch.sin(fx * 3.14159 * xx + fy * 3.14159 * yy + ph)
        return acc / (acc.abs().max() + 1e-6)

    return torch.stack([chan(), chan(), chan()])[None]


def _trace_example_image():
    """A NON-DEGENERATE [-1,1] example image for tracing encode. Degenerate examples (zeros, or
    smooth low-entropy synthetics) make torch.export/coreai specialize a graph that is only correct
    near that regime and produces a catastrophically wrong KV cache for real camera images. A real
    photo (or a high-entropy natural-statistics fallback) traces the correct, input-general graph.
    Set SMOLVLA_TRACE_IMAGE to a JPEG/PNG path to override the built-in fallback."""
    import torch
    p = os.environ.get("SMOLVLA_TRACE_IMAGE")
    if not p:
        for cand in ("build/_realimg/img0.bin", "build/_realimg/img0.jpg"):
            if Path(cand).exists():
                p = cand
                break
    if p and Path(p).exists():
        import numpy as np
        from PIL import Image
        # THREE DISTINCT crops (the 3 camera slots must be traced with distinct images — tracing
        # all-3-identical falls on the same degenerate specialization as zeros).
        base = Path(p).parent
        cands = sorted(base.glob("img*.bin")) + sorted(base.glob("img*.jpg")) if base.exists() else []
        cands = [str(c) for c in cands] or [p, p, p]
        g = torch.Generator().manual_seed(205)
        outs = []
        for j in range(3):
            im = Image.open(cands[j % len(cands)]).convert("RGB")
            w, h = im.size
            scale = 0.6 + 0.4 * torch.rand(1, generator=g).item()
            cw = int(min(w, h) * scale)
            x = int((w - cw) * torch.rand(1, generator=g).item())
            y = int((h - cw) * torch.rand(1, generator=g).item())
            im = im.crop((x, y, x + cw, y + cw)).resize((IMG, IMG), Image.BILINEAR)
            arr = np.asarray(im).astype("float32")
            outs.append(torch.from_numpy(arr).permute(2, 0, 1)[None] / 127.5 - 1.0)
        return outs
    # Fallback: 3 DISTINCT high-entropy natural-statistics images (self-contained, no photo needed).
    g = torch.Generator().manual_seed(0)
    return [natural_image(g) for _ in range(3)]


def cmd_export(out: Path, fp16: bool = True, free_weights: bool = False):
    import torch
    enc, den, d = _build_wrappers(fp16=fp16)
    if free_weights:
        sf = Path(os.environ.get("SMOLVLA_CONFIG_DIR", "build/_hf_mirror/smolvla")) / "model.safetensors"
        if sf.exists():
            sf.unlink()
            print(f"freed {sf} — {_disk_free_gb():.1f}GB free")
    out.mkdir(parents=True, exist_ok=True)
    # Trace encode with NON-DEGENERATE example inputs. All-zero dummy images make
    # torch.export(strict=False) bake input-dependent specialization that is only correct near
    # zero: the lowered encode then yields a plausible KV cache for small/synthetic activations
    # but a CATASTROPHICALLY wrong one for real (large-activation) camera images (real-image
    # action cosine 0.02 vs 0.99). A structured, in-range [-1,1] example image + non-zero state
    # forces a correct, input-general graph. (denoise's cache example stays zeros — it is a plain
    # transformer over the cache and does not specialize the same way.)
    ex = [t.to(d["img"].dtype) for t in _trace_example_image()]
    ex_state = (torch.randn(1, STATE_DIM, generator=torch.Generator().manual_seed(0)) * 0.1).to(d["state"].dtype)
    enc_ep = torch.export.export(
        enc, args=(ex[0], ex[1], ex[2], d["imask"], d["imask"], d["imask"],
                   d["tok"], d["lmask"], ex_state), strict=False)
    torch.export.save(enc_ep, str(out / "encode.pt2"))
    del enc_ep
    print(f"ok: wrote {out}/encode.pt2 (prefix_len={PREFIX_LEN}) — {_disk_free_gb():.1f}GB free")
    with no_deepcopy():
        den_ep = torch.export.export(
            den, args=(d["ppad"], d["xt"], d["t"], *d["cache"]), strict=False)
    torch.export.save(den_ep, str(out / "denoise_step.pt2"))
    print(f"ok: wrote {out}/denoise_step.pt2 — {_disk_free_gb():.1f}GB free")
    print("next (venv-B): .venv/bin/python models/smolvla/export.py --lower --out", out)


def cmd_lower(out: Path):
    import torch
    from coreai_torch import TorchConverter, get_decomp_table

    # coreai_torch has no aten.empty_permuted lowering. It comes from pixel_shuffle's
    # permute().reshape() (SmolVLMConnector): the reshape-after-permute needs a contiguous copy in
    # the PERMUTED memory order. A plain zeros(size) has the wrong strides → the copy/reshape
    # garbles the image features → a wrong KV cache (localized: denoise-with-torch-cache = 1.0, full
    # = 0.99). Build the buffer in physical order then permute to the logical layout so strides match.
    def _empty_permuted_decomp(size, physical_layout, dtype=None, layout=None, device=None,
                               pin_memory=None):
        phys = [size[i] for i in physical_layout]
        inv = [0] * len(physical_layout)
        for i, p in enumerate(physical_layout):
            inv[p] = i
        return torch.zeros(phys, dtype=dtype, device=device).permute(*inv)

    def _load(name):
        ep = torch.export.load(str(out / f"{name}.pt2"))
        table = dict(get_decomp_table())
        table[torch.ops.aten.empty_permuted.default] = _empty_permuted_decomp
        return ep.run_decompositions(table)

    conv = TorchConverter()
    conv.add_exported_program(_load("encode"), input_names=_ENC_INPUTS,
                              output_names=_ENC_OUTPUTS, entrypoint_name="encode")
    conv.add_exported_program(_load("denoise_step"), input_names=_DEN_INPUTS,
                              output_names=_DEN_OUTPUTS, entrypoint_name="denoise_step")
    prog = conv.to_coreai()
    for name in ("encode", "denoise_step"):
        p = out / f"{name}.pt2"
        if p.exists():
            p.unlink()
    print(f"freed .pt2 — {_disk_free_gb():.1f}GB free")
    prog.optimize()
    aimodel = out / f"{out.name}.aimodel"
    prog.save_asset(aimodel)
    print(f"ok: lowered SmolVLA -> {aimodel} (encode, denoise_step) — {_disk_free_gb():.1f}GB free")


def main():
    ap = argparse.ArgumentParser(description="SmolVLA split-export")
    ap.add_argument("phase", nargs="?", default="export", choices=["export"])
    ap.add_argument("--lower", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--free-weights", action="store_true")
    args = ap.parse_args()
    if args.lower:
        cmd_lower(args.out)
    else:
        cmd_export(args.out, fp16=not args.fp32, free_weights=args.free_weights)


if __name__ == "__main__":
    main()
