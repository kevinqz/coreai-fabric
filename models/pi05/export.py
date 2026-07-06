# /// script
# requires-python = ">=3.12"
# ///
"""pi0 (LeRobot flow-matching VLA) -> Apple .aimodel, split-export.

Per-model export script — fabric RECORDS + PRINTS this invocation but does NOT
auto-drive it (like models/depth-anything/export.py; recipe pi0-base.yaml declares
`conversion.tool: models/pi0/export.py`). You run it by hand, in two venvs.

SPLIT (VERIFIED against lerobot modeling_pi0.py PaliGemmaWithExpertModel.forward,
a 3-branch fn keyed on which of inputs_embeds=[prefix,suffix] is None):
  - `encode`      : VLM prefix (SigLIP x3 cams + Gemma-2b) -> prefix KV cache. Run ONCE per obs.
  - `denoise_step`: action expert (Gemma-300m, width 1024) consuming the KV cache -> velocity v_t.
                    The host drives this num_steps(=10) times (Euler flow-matching).

TWO-PHASE, TWO-VENV:
  export  (venv-A, .venv-lerobot, torch 2.9 + lerobot[pi]): torch.export -> encode.pt2 / denoise_step.pt2
  --lower (venv-B, fabric .venv,  torch 2.9 + coreai_torch): load the .pt2 -> lower -> <bundle>/encode|denoise_step/main.*

Usage:
  # venv-A:
  .venv-lerobot/bin/python models/pi0/export.py export --out build/pi0-base
  # venv-B:
  .venv/bin/python         models/pi0/export.py --lower --out build/pi0-base

TWO SOURCE EDITS applied at trace time (both VERIFIED export-hostile in modeling_pi0.py):
  1. denoise_step does copy.deepcopy(past_key_values) (~:918) — the host owns the cache in
     the split; we monkeypatch a deepcopy-free path (the cache arrives as static tensors).
  2. keep _attn_implementation="eager" (the model already forces it) — SDPA/flash don't
     trace cleanly, and eager feeds coreai_torch's replace_sdpa a DENSE additive float mask
     (this is the documented bypass for the coremltools Gemma-3 __ior__ mask blocker #2560).

GATE 2a (shape, UNCERTAIN until probed): prefix_len == num_img_embs*3 + 48. num_img_embs is
  256 (SigLIP patch tokens) -> 816, OR 1 (pooled) -> 51. RESOLVE with pi0_export_probe.py
  (it prints the real prefix_len) and set PI0_PREFIX_LEN before a full export. Default 816.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# pi05 shape varies by checkpoint: folding=chunk30/TOK48, LIBERO/base=chunk50/TOK200 (the
# proprioceptive state is discretized INTO the language tokens -> a longer prompt), droid=chunk15.
# Set PI05_CHUNK + PI05_TOK to the checkpoint's chunk_size + tokenizer_max_length before export.
CHUNK, ACT_DIM, STATE_DIM, TOK = (int(os.environ.get("PI05_CHUNK", "30")), 32, 32,
                                  int(os.environ.get("PI05_TOK", "48")))
# prefix = 3 cams * 256 SigLIP patch embeds + TOK language tokens (probe confirms num_img_embs=256).
# Derived from TOK so it tracks the prompt length; override PI0_PREFIX_LEN only if a checkpoint differs.
PREFIX_LEN = int(os.environ.get("PI0_PREFIX_LEN", str(3 * 256 + TOK)))
N_LAYERS, KV_HEADS, HEAD_DIM = 18, 1, 256                    # gemma_2b prefix cache dims (VERIFIED)


import contextlib


def _safe_make_att_2d_masks(pad_masks, att_masks):
    """Bool-mul-free make_att_2d_masks. The upstream does `pad_masks * pad_masks` on BOOL
    tensors — it lowers, but the coreai runtime has no bool `mul` kernel (KernelError
    unsupportedType bool at inference). Recompute with int32 mul + a comparison (supported
    kernels); identical result (a bool 2-D attention mask). Used by BOTH encode (the wrapper)
    and denoise (denoise_step calls it internally), so it is monkeypatched module-wide."""
    import torch
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = (cumsum[:, None, :] <= cumsum[:, :, None]).to(torch.int32)
    pad = pad_masks.to(torch.int32)
    pad_2d = pad[:, None, :] * pad[:, :, None]
    return (att_2d * pad_2d) > 0                      # int mul + compare -> bool, no bool mul/and


@contextlib.contextmanager
def no_deepcopy():
    """denoise_step deepcopies the KV cache internally (lerobot ~:918); deepcopy fails
    on a FakeTensor during torch.export ("Cannot access data pointer"). Identity-copy is
    safe here — we export a SINGLE step and drive it with fresh inputs, never mutating a
    shared cache. Runbook edit #1."""
    import copy as _copy
    orig = _copy.deepcopy
    _copy.deepcopy = lambda x, memo=None: x
    try:
        yield
    finally:
        _copy.deepcopy = orig


# lerobot 0.5.1's PI0Config REJECTS these checkpoint fields (older-pi0 config drift).
# All 8 are vestigial/runtime — grep-verified UNUSED in 0.5.1's pi0 code — so stripping
# them keeps the architecture 0.5.1 builds identical to the checkpoint's state_dict.
_DRIFT_FIELDS = ("num_steps", "proj_width", "attention_implementation", "train_state_proj",
                 "resize_imgs_with_padding", "adapt_to_pi_aloha",
                 "use_delta_joint_actions_aloha", "use_cache")


def _load_pi0_config(cfg_dir: str):
    """Load the pi05 config for lerobot 0.5.1's STRICT draccus decode (rejects ANY key not on
    PI05Config). Different checkpoints carry different legacy/plumbing keys: folding is clean,
    but the LIBERO/base pi05 checkpoints carry ~20 extra keys (paligemma_variant, time_sampling_*,
    compile_*, repo_id, ...) that ARE valid PI05Config fields in 0.5.1. Keep only keys PI05Config
    declares (introspected from the dataclass) + 'type' — robust to any drift (mirrors models/pi0)."""
    import json
    import tempfile
    import dataclasses
    from pathlib import Path as _P
    import lerobot.policies.pi05.modeling_pi05  # noqa: F401 — registers the 'pi05' draccus choice class
    from lerobot.policies.pi05.modeling_pi05 import PI05Config
    from lerobot.configs.policies import PreTrainedConfig
    raw = json.loads((_P(cfg_dir) / "config.json").read_text())
    valid = {f.name for f in dataclasses.fields(PI05Config)} | {"type"}
    stripped = {k: v for k, v in raw.items() if k in valid}
    dropped = sorted(set(raw) - set(stripped))
    if dropped:
        print(f"pi05 config: dropped {len(dropped)} non-PI05Config key(s) for strict decode: {dropped}")
    tmp = _P(tempfile.mkdtemp(prefix="pi05cfg_"))
    (tmp / "config.json").write_text(json.dumps(stripped))
    return PreTrainedConfig.from_pretrained(str(tmp), local_files_only=True)


def _load_policy(fp16: bool = False):
    """venv-A. PI0_RANDOM_INIT=1 -> random weights from config (no download, op-coverage).
    Else load REAL weights from PI0_CONFIG_DIR (a local mirror) with the drift-stripped
    config passed in, so from_pretrained skips its own strict config parse."""
    import os
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    cfg_dir = os.environ.get("PI0_CONFIG_DIR", "build/_hf_mirror/folding_latest")
    cfg = _load_pi0_config(cfg_dir)
    if os.environ.get("PI0_RANDOM_INIT") == "1":
        policy = PI05Policy(cfg).eval()                     # random weights — op-coverage only
    else:
        # Pass the stripped config in so from_pretrained loads the real state_dict from
        # the local mirror without re-parsing (and rejecting) the drifted config.json.
        policy = PI05Policy.from_pretrained(cfg_dir, config=cfg, local_files_only=True).eval()
    policy = policy.to("cpu")   # pi05's config device=cuda->mps leaks the vision tower onto MPS
    if fp16:
        policy = policy.half()
        # fp16 gotcha: the gemma action expert upcasts internally and hands action_out_proj
        # an fp32 tensor while the proj weight is fp16 -> "mat1 Float / mat2 Half". Cast each
        # projection's input to its own weight dtype via a forward-pre-hook (traceable through
        # torch.export). Covers the out/in/state projections defensively.
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


def _build_wrappers(fp16: bool = False):
    """venv-A only. Returns (encode_module, denoise_module, dummy_args_dict)."""
    import torch
    from lerobot.policies.pi05 import modeling_pi05 as _mp

    # Install the bool-mul-free mask builder module-wide: the encode wrapper calls it below
    # AND denoise_step calls the module global internally — both must avoid the bool `mul`.
    _mp.make_att_2d_masks = _safe_make_att_2d_masks

    policy = _load_policy(fp16=fp16)
    m = policy.model  # PI05Pytorch

    class EncodeWrapper(torch.nn.Module):
        def forward(self, img0, img1, img2, imask0, imask1, imask2, lang_tokens, lang_masks):
            pe, ppad, patt = m.embed_prefix([img0, img1, img2], [imask0, imask1, imask2],
                                            lang_tokens, lang_masks)
            att2d = _safe_make_att_2d_masks(ppad, patt)
            pos = torch.cumsum(ppad, 1) - 1
            att4d = m._prepare_attention_masks_4d(att2d)
            _, pkv = m.paligemma_with_expert.forward(
                attention_mask=att4d, position_ids=pos, past_key_values=None,
                inputs_embeds=[pe, None], use_cache=True)
            return (ppad, *_flatten_cache(pkv))

    class DenoiseWrapper(torch.nn.Module):
        def forward(self, prefix_pad_masks, x_t, timestep, *cache_tensors):
            pkv = _unflatten_cache(cache_tensors)
            # pi05 denoise_step drops `state` (no proprioceptive input; state handled via prefix/
            # adarms). deepcopy of the KV cache is neutralized by no_deepcopy() around the export.
            return m.denoise_step(prefix_pad_masks=prefix_pad_masks,
                                  past_key_values=pkv, x_t=x_t, timestep=timestep)

    import torch as _t
    fd = _t.float16 if fp16 else _t.float32   # float example inputs must match the model dtype
    d = dict(
        img=_t.zeros(1, 3, 224, 224, dtype=fd), imask=_t.ones(1, dtype=_t.bool),
        tok=_t.zeros(1, TOK, dtype=_t.long), lmask=_t.ones(1, TOK, dtype=_t.bool),
        state=_t.zeros(1, STATE_DIM, dtype=fd), ppad=_t.ones(1, PREFIX_LEN, dtype=_t.bool),
        xt=_t.zeros(1, CHUNK, ACT_DIM, dtype=fd), t=_t.zeros(1, dtype=fd),
        cache=tuple(_t.zeros(1, KV_HEADS, PREFIX_LEN, HEAD_DIM, dtype=fd) for _ in range(2 * N_LAYERS)),
    )
    return EncodeWrapper().eval(), DenoiseWrapper().eval(), d


def _flatten_cache(pkv):
    out = []
    for layer in pkv:                       # HF cache -> plain tensors (k,v per layer)
        out.extend([layer[0], layer[1]])
    return tuple(out)


def _unflatten_cache(tensors):
    # lerobot 0.5.1's pi_gemma/paligemma expert expects a transformers Cache (it calls
    # .get_seq_length()), not a legacy list. Rebuild a DynamicCache from the crossed-in
    # KV tensors (the split-export boundary carries the cache as plain tensors).
    from transformers.cache_utils import DynamicCache
    legacy = tuple((tensors[2 * i], tensors[2 * i + 1]) for i in range(len(tensors) // 2))
    return DynamicCache(legacy)   # transformers 5.x: ctor takes the per-layer (k,v) iterable


#: consistent tensor names across the encode->denoise KV-cache boundary + the .aimodel I/O.
_CACHE_NAMES = [f"cache_{i}" for i in range(2 * N_LAYERS)]         # 36 = k,v per 18 layers
_ENC_INPUTS = ["img0", "img1", "img2", "imask0", "imask1", "imask2", "lang_tokens", "lang_masks"]
_ENC_OUTPUTS = ["prefix_pad_masks", *_CACHE_NAMES]
_DEN_INPUTS = ["prefix_pad_masks", "x_t", "timestep", *_CACHE_NAMES]
_DEN_OUTPUTS = ["velocity"]


def cmd_export(out: Path, fp16: bool = True, free_weights: bool = False):
    """venv-A: trace both graphs -> .pt2. fp16 halves the ~11.7GB encode.pt2 (fits ~30GB disk).
    free_weights deletes the 14GB safetensors right after load (model is in RAM) to make room."""
    import os
    import torch
    enc, den, d = _build_wrappers(fp16=fp16)
    if free_weights:
        sf = Path(os.environ.get("PI0_CONFIG_DIR", "build/_hf_mirror/pi0")) / "model.safetensors"
        if sf.exists():
            sf.unlink()
            print(f"freed {sf} (model is resident in RAM) — {_disk_free_gb():.1f}GB free")
    out.mkdir(parents=True, exist_ok=True)
    enc_ep = torch.export.export(
        enc, args=(d["img"], d["img"], d["img"], d["imask"], d["imask"], d["imask"],
                   d["tok"], d["lmask"]), strict=False)
    torch.export.save(enc_ep, str(out / "encode.pt2"))
    del enc_ep
    print(f"ok: wrote {out}/encode.pt2 ({'fp16' if fp16 else 'fp32'}, prefix_len={PREFIX_LEN}) — "
          f"{_disk_free_gb():.1f}GB free")
    with no_deepcopy():
        den_ep = torch.export.export(
            den, args=(d["ppad"], d["xt"], d["t"], *d["cache"]), strict=False)
    torch.export.save(den_ep, str(out / "denoise_step.pt2"))
    print(f"ok: wrote {out}/denoise_step.pt2 — {_disk_free_gb():.1f}GB free")
    print("next (venv-B): .venv/bin/python models/pi0/export.py --lower --out", out)


def _disk_free_gb() -> float:
    import shutil
    return shutil.disk_usage("/").free / 1e9


def cmd_lower(out: Path):
    """venv-B: load both .pt2 -> ONE .aimodel with encode + denoise_step entrypoints (the
    action_parity split contract). Deletes each .pt2 after adding it, to stay under ~30GB disk."""
    import torch
    from coreai_torch import TorchConverter, get_decomp_table

    def _load(name):
        ep = torch.export.load(str(out / f"{name}.pt2"))
        return ep.run_decompositions(get_decomp_table())

    conv = TorchConverter()
    conv.add_exported_program(_load("encode"), input_names=_ENC_INPUTS,
                              output_names=_ENC_OUTPUTS, entrypoint_name="encode")
    conv.add_exported_program(_load("denoise_step"), input_names=_DEN_INPUTS,
                              output_names=_DEN_OUTPUTS, entrypoint_name="denoise_step")
    prog = conv.to_coreai()
    # to_coreai has fully consumed both staged ExportedPrograms — free the big .pt2 (up to
    # ~13GB) BEFORE optimize()/save_asset writes the .aimodel, so the two never coexist on
    # this ~95%-full disk.
    for name in ("encode", "denoise_step"):
        p = out / f"{name}.pt2"
        if p.exists():
            p.unlink()
    print(f"freed encode/denoise .pt2 — {_disk_free_gb():.1f}GB free")
    prog.optimize()
    aimodel = out / f"{out.name}.aimodel"
    prog.save_asset(aimodel)   # MUST be a Path ending in .aimodel (save_asset checks .suffix)
    print(f"ok: lowered pi0 -> {aimodel} (entrypoints: encode, denoise_step) — "
          f"{_disk_free_gb():.1f}GB free")
    print("next: write norm_stats.json (from policy_pre/postprocessor), then models/pi0/parity.py")


def main():
    ap = argparse.ArgumentParser(description="pi0 split-export (see docs/vla-export-runbook.md)")
    ap.add_argument("phase", nargs="?", default="export", choices=["export"])
    ap.add_argument("--lower", action="store_true", help="venv-B: lower the .pt2s to .aimodel")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fp32", action="store_true", help="export fp32 (default fp16 to fit disk)")
    ap.add_argument("--free-weights", action="store_true",
                    help="delete model.safetensors after load to reclaim ~14GB disk")
    args = ap.parse_args()
    if args.lower:
        cmd_lower(args.out)
    else:
        cmd_export(args.out, fp16=not args.fp32, free_weights=args.free_weights)


if __name__ == "__main__":
    main()
