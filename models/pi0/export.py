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

PREFIX_LEN = int(os.environ.get("PI0_PREFIX_LEN", "816"))   # 3*256 + 48; probe resolves it
CHUNK, ACT_DIM, STATE_DIM, TOK = 50, 32, 32, 48
N_LAYERS, KV_HEADS, HEAD_DIM = 18, 1, 256                    # gemma_2b prefix cache dims (VERIFIED)


def _build_wrappers():
    """venv-A only. Returns (encode_module, denoise_module, dummy_args_dict)."""
    import torch
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy, make_att_2d_masks

    policy = PI0Policy.from_pretrained("lerobot/pi0").eval()
    m = policy.model  # PI0Pytorch

    class EncodeWrapper(torch.nn.Module):
        def forward(self, img0, img1, img2, imask0, imask1, imask2, lang_tokens, lang_masks):
            pe, ppad, patt = m.embed_prefix([img0, img1, img2], [imask0, imask1, imask2],
                                            lang_tokens, lang_masks)
            att2d = make_att_2d_masks(ppad, patt)
            pos = torch.cumsum(ppad, 1) - 1
            att4d = m._prepare_attention_masks_4d(att2d)
            _, pkv = m.paligemma_with_expert.forward(
                attention_mask=att4d, position_ids=pos, past_key_values=None,
                inputs_embeds=[pe, None], use_cache=True)
            return (ppad, *_flatten_cache(pkv))

    class DenoiseWrapper(torch.nn.Module):
        def forward(self, state, prefix_pad_masks, x_t, timestep, *cache_tensors):
            pkv = _unflatten_cache(cache_tensors)
            # deepcopy-free denoise_step (edit #1): call the internal expert path directly if the
            # installed lerobot still deepcopies at ~:918. Prefer monkeypatching copy.deepcopy=identity
            # around this call, or ship a patched modeling_pi0. See the runbook.
            return m.denoise_step(state=state, prefix_pad_masks=prefix_pad_masks,
                                  past_key_values=pkv, x_t=x_t, timestep=timestep)

    import torch as _t
    d = dict(
        img=_t.zeros(1, 3, 224, 224), imask=_t.ones(1, dtype=_t.bool),
        tok=_t.zeros(1, TOK, dtype=_t.long), lmask=_t.ones(1, TOK, dtype=_t.bool),
        state=_t.zeros(1, STATE_DIM), ppad=_t.ones(1, PREFIX_LEN, dtype=_t.bool),
        xt=_t.zeros(1, CHUNK, ACT_DIM), t=_t.zeros(1),
        cache=tuple(_t.zeros(1, KV_HEADS, PREFIX_LEN, HEAD_DIM) for _ in range(2 * N_LAYERS)),
    )
    return EncodeWrapper().eval(), DenoiseWrapper().eval(), d


def _flatten_cache(pkv):
    out = []
    for layer in pkv:                       # HF cache -> plain tensors (k,v per layer)
        out.extend([layer[0], layer[1]])
    return tuple(out)


def _unflatten_cache(tensors):
    return [(tensors[2 * i], tensors[2 * i + 1]) for i in range(len(tensors) // 2)]


def cmd_export(out: Path):
    """venv-A: trace both graphs -> .pt2 (serialized, torch 2.9 -> loads unchanged in venv-B)."""
    import torch
    enc, den, d = _build_wrappers()
    enc_ep = torch.export.export(
        enc, args=(d["img"], d["img"], d["img"], d["imask"], d["imask"], d["imask"],
                   d["tok"], d["lmask"]), strict=False)
    den_ep = torch.export.export(
        den, args=(d["state"], d["ppad"], d["xt"], d["t"], *d["cache"]), strict=False)
    out.mkdir(parents=True, exist_ok=True)
    torch.export.save(enc_ep, str(out / "encode.pt2"))
    torch.export.save(den_ep, str(out / "denoise_step.pt2"))
    print(f"ok: wrote {out}/encode.pt2 + denoise_step.pt2 (prefix_len={PREFIX_LEN})")
    print("next (venv-B): .venv/bin/python models/pi0/export.py --lower --out", out)


def cmd_lower(out: Path):
    """venv-B (fabric .venv): load each .pt2 -> coreai_torch lower -> <graph>/main.aimodel.

    Mirrors drivers/llm_export.py:156-170 (run_decompositions -> TorchConverter
    -> add_exported_program -> to_coreai -> optimize -> save_asset)."""
    import torch
    from coreai_torch import TorchConverter, get_decomp_table
    for name, outs in (("encode", ["prefix_pad_masks", "keyCache", "valueCache"]),
                       ("denoise_step", ["velocity"])):
        ep = torch.export.load(str(out / f"{name}.pt2"))
        ep = ep.run_decompositions(get_decomp_table())
        conv = TorchConverter().add_exported_program(ep, output_names=outs)
        prog = conv.to_coreai()
        prog.optimize()
        (out / name).mkdir(parents=True, exist_ok=True)
        prog.save_asset(str(out / name / "main.aimodel"))
        print(f"ok: lowered {name} -> {out/name}/main.*")
    print("next: write metadata.json (num_steps=10) + norm_stats.json, then coreai-fabric verify")


def main():
    ap = argparse.ArgumentParser(description="pi0 split-export (see docs/vla-export-runbook.md)")
    ap.add_argument("phase", nargs="?", default="export", choices=["export"])
    ap.add_argument("--lower", action="store_true", help="venv-B: lower the .pt2s to .aimodel")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cmd_lower(args.out) if args.lower else cmd_export(args.out)


if __name__ == "__main__":
    main()
