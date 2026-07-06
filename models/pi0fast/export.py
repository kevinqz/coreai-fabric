"""pi0fast (LeRobot pi0_fast, AUTOREGRESSIVE VLA) -> Apple .aimodel, split-export.

pi0fast = ONE PaliGemma-2B causal LM (no action expert, no flow-matching). Actions become discrete
tokens via the FAST tokenizer (host-side DCT+BPE), the gemma decoder generates them autoregressively,
the host FAST-detokenizes. Split (analogous to pi0 encode/denoise + the VLM main):
  - `encode`      : SigLIP x3 + gemma prefix (image 256 x3 + lang 200 + BOS) -> KV cache + first_logits.
  - `decode_step` : one AR gemma step (embed token x sqrt(width) -> forward w/ cache -> lm_head) -> next
                    logits + updated cache. Host drives the fixed 256-step greedy argmax loop.

STANDARD RoPE (gemma_2b, no M-RoPE) so no VLM-style greedy cap. The PaliGemma text tokenizer
(google/paligemma-3b-pt-224) is GATED; the graph only needs bos_token_id=2, so we STUB it (the FAST
tokenizer, lerobot/fast-action-tokenizer, is ungated and only used host-side for detok).

OPEN CHALLENGE (the riskiest step): the crossed-tensor KV cache GROWS during decode (969 -> 1225),
unlike pi0's fixed-prefix denoise. coreai wants static shapes. cmd_lower must pin the cache to a
fixed MAX_TOTAL via a StaticCache (write-at-cache_position) OR coreai dynamic shapes on the seq dim.
This file traces with DynamicCache (correct torch reference); the fixed-max reformulation is applied
at lower time — see cmd_lower TODO.

  export  (venv-A, .venv-lerobot): torch.export -> encode.pt2 / decode_step.pt2
  --lower (venv-B, .venv + coreai_torch): -> ONE .aimodel (encode + decode_step)
"""
import argparse
import os
from pathlib import Path

# Probed on lerobot/pi0fast-base (config gemma_2b, tok_max 200): PREFIX_LEN = 3*256 img + 200 lang + 1 BOS.
IMG_EMBS = 256                                   # PaliGemma get_image_features patch tokens per camera
TOK = int(os.environ.get("PI0FAST_TOK", "200"))  # tokenizer_max_length
PREFIX_LEN = int(os.environ.get("PI0FAST_PREFIX_LEN", str(3 * IMG_EMBS + TOK + 1)))  # 969
N_LAYERS, KV_HEADS, HEAD_DIM = 18, 1, 256        # gemma_2b prefix cache dims (VERIFIED, = pi0)
VOCAB, WIDTH = 257152, 2048
BOS = 2                                           # gemma bos_token_id (stubbed; avoids the gated tokenizer)
MAX_DECODE = int(os.environ.get("PI0FAST_MAX_DECODE", "256"))
MAX_TOTAL = PREFIX_LEN + MAX_DECODE               # 1225 — fixed-max cache length for the AR decode


class _StubTok:
    """bos=2 stand-in for the GATED google/paligemma-3b-pt-224 tokenizer. The graph only needs
    bos_token_id; host-side detok uses the ungated FAST tokenizer + the id<->act-id formula."""
    bos_token_id, eos_token_id, pad_token_id, vocab_size = 2, 1, 0, VOCAB

    def convert_ids_to_tokens(self, ids):
        return [str(int(i)) for i in ids]


def _install_tok_stub():
    import lerobot.policies.pi0_fast.modeling_pi0_fast as _mp
    orig = _mp.AutoTokenizer.from_pretrained
    _mp.AutoTokenizer.from_pretrained = staticmethod(
        lambda name, *a, **k: _StubTok() if "paligemma" in str(name).lower() else orig(name, *a, **k))


def _load_pi0fast_config(cfg_dir: str):
    """Allowlist the config to PI0FastConfig's declared fields (strict draccus decode)."""
    import json
    import tempfile
    import dataclasses
    from pathlib import Path as _P
    import lerobot.policies.pi0_fast.modeling_pi0_fast  # noqa: F401 — registers 'pi0_fast'
    from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
    from lerobot.configs.policies import PreTrainedConfig
    raw = json.loads((_P(cfg_dir) / "config.json").read_text())
    valid = {f.name for f in dataclasses.fields(PI0FastConfig)} | {"type"}
    tmp = _P(tempfile.mkdtemp(prefix="pi0fastcfg_"))
    (tmp / "config.json").write_text(json.dumps({k: v for k, v in raw.items() if k in valid}))
    return PreTrainedConfig.from_pretrained(str(tmp), local_files_only=True)


def _load_policy(fp16: bool = False):
    """venv-A. PI0FAST_RANDOM_INIT=1 -> random weights (op-coverage, no download). Pins to CPU (the
    config device=cuda->mps leaks the SigLIP tower onto MPS). Stubs the gated PaliGemma tokenizer."""
    _install_tok_stub()
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
    cfg_dir = os.environ.get("PI0_CONFIG_DIR", "build/_hf_mirror/pi0fast")
    if os.environ.get("PI0FAST_RANDOM_INIT") == "1":
        cfg = _load_pi0fast_config_from_hub()
        policy = PI0FastPolicy(cfg).eval()
    else:
        cfg = _load_pi0fast_config(cfg_dir)
        policy = PI0FastPolicy.from_pretrained(cfg_dir, config=cfg, local_files_only=True).eval()
    policy = policy.to("cpu")
    if fp16:
        policy = policy.half()
    return policy


def _load_pi0fast_config_from_hub():
    import json
    import tempfile
    import dataclasses
    from pathlib import Path as _P
    from huggingface_hub import hf_hub_download
    import lerobot.policies.pi0_fast.modeling_pi0_fast  # noqa: F401
    from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
    from lerobot.configs.policies import PreTrainedConfig
    raw = json.loads(_P(hf_hub_download("lerobot/pi0fast-base", "config.json")).read_text())
    valid = {f.name for f in dataclasses.fields(PI0FastConfig)} | {"type"}
    tmp = _P(tempfile.mkdtemp(prefix="pi0fastcfg_"))
    (tmp / "config.json").write_text(json.dumps({k: v for k, v in raw.items() if k in valid}))
    return PreTrainedConfig.from_pretrained(str(tmp), local_files_only=True)


def _flatten_cache(pkv):
    out = []
    for layer in pkv:
        out.extend([layer[0], layer[1]])
    return tuple(out)


def _unflatten_cache(tensors):
    from transformers.cache_utils import DynamicCache
    legacy = tuple((tensors[2 * i], tensors[2 * i + 1]) for i in range(len(tensors) // 2))
    return DynamicCache(legacy)


def natural_image(g, size=224):
    """3-distinct-image [-1,1] tracing example. pi0fast's `_preprocess_images` does the [0,1]->[-1,1]
    SigLIP normalization OUTSIDE the graph (modeling_pi0_fast.py:1082), so `embed_prefix_fast` — and
    therefore the encode graph and the parity obs — consume images already in [-1,1] (like SmolVLA,
    NOT like pi0's [0,1]). Distinct + high-entropy 1/f-ish avoids the degenerate-trace collapse."""
    import torch
    lin = torch.linspace(-1, 1, size)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")

    def chan():
        acc = torch.zeros(size, size)
        for _ in range(40):
            fx, fy = (torch.rand(2, generator=g) * 16 + 0.5).tolist()
            ph = torch.rand(1, generator=g).item() * 6.283
            acc = acc + (1.0 / (abs(fx) + abs(fy) + 1.0)) * torch.sin(fx * 3.14159 * xx + fy * 3.14159 * yy + ph)
        return acc / (acc.abs().max() + 1e-6)

    return torch.stack([chan(), chan(), chan()])[None]  # [-1,1], deployment/SigLIP range


# StaticCache DEPLOYABLE decode (O(1)/step, ~10x faster than the O(n^2) recompute — 10s vs 97s/token
# on-device). Fixed-max [1,KV_HEADS,MAX_TOTAL,HEAD_DIM] KV cache carried as plain tensors across the
# encode->decode boundary (like pi0's crossed-tensor cache, but AUTOREGRESSIVE): `encode` PREFILLS the
# cache (prefix forward, cache_position=arange(PREFIX_LEN)) + emits first_logits; `decode_step` forwards
# ONE token, writes its K/V at cache_position via a functional index_copy, attending to the valid span
# through a host-built [1,1,1,MAX_TOTAL] additive mask. Custom functional TensorCache because
# transformers' StaticCache uses lazy in-place object buffers (not exportable as tensor I/O); stock
# GemmaAttention uses update()'s RETURN, so the functional cache drops in. `_patch_forward` threads
# cache_position (PI0FastPaliGemma.forward drops it). Encode also avoids _create_custom_attention_mask_fast's
# bool*bool (modeling_pi0_fast.py:488) which the coreai runtime rejects — masks are host/graph-built floats.
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38
_CACHE_K = [f"k_{i}" for i in range(N_LAYERS)]
_CACHE_V = [f"v_{i}" for i in range(N_LAYERS)]
_ENC_INPUTS = ["img0", "img1", "img2", "lang_tokens"]
_ENC_OUTPUTS = ["first_logits", *_CACHE_K, *_CACHE_V]
_DEN_INPUTS = ["new_token", "att_4d", "position_id", "cache_position", *_CACHE_K, *_CACHE_V]
_DEN_OUTPUTS = ["logits", *_CACHE_K, *_CACHE_V]


class TensorCache:
    """Functional fixed-max KV cache: update() index_copy's the new K/V at cache_position and returns
    the FULL [1,KV_HEADS,MAX_TOTAL,HEAD_DIM] buffers (stock GemmaAttention uses the return). No in-place
    object mutation, so it exports cleanly to coreai as plain tensor I/O (36 = k,v per 18 layers)."""
    def __init__(self, keys, values):
        self.keys, self.values = list(keys), list(values)

    def update(self, key, value, layer_idx, cache_kwargs=None):
        import torch
        pos = cache_kwargs["cache_position"]
        self.keys[layer_idx] = torch.index_copy(self.keys[layer_idx], 2, pos, key.to(self.keys[layer_idx].dtype))
        self.values[layer_idx] = torch.index_copy(self.values[layer_idx], 2, pos, value.to(self.values[layer_idx].dtype))
        return self.keys[layer_idx], self.values[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return MAX_TOTAL                      # cache_position is always passed explicitly; keep static

    def flatten(self):
        return (*self.keys, *self.values)


def _patch_forward(m):
    """PI0FastPaliGemma.forward drops cache_position; thread it through to language_model.forward so a
    StaticCache can write-at-position."""
    pwe = m.paligemma_with_expert
    orig_lm = pwe.paligemma.model.language_model.forward

    def fwd(attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
            use_cache=None, adarms_cond=None, cache_position=None):
        out = orig_lm(inputs_embeds=inputs_embeds[0], attention_mask=attention_mask,
                      position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache,
                      cache_position=cache_position,
                      adarms_cond=adarms_cond[0] if adarms_cond is not None else None)
        return [out.last_hidden_state, None], out.past_key_values

    pwe.forward = fwd


def _build_wrappers(fp16: bool = False):
    """venv-A only. StaticCache encode (prefill) + decode_step (O(1) write-at-position)."""
    import math
    import torch
    policy = _load_policy(fp16=fp16)
    m = policy.model
    _patch_forward(m)
    pwe = m.paligemma_with_expert
    lm_head = pwe.paligemma.lm_head
    fd = torch.float16 if fp16 else torch.float32

    def _zero_cache():
        return TensorCache([torch.zeros(1, KV_HEADS, MAX_TOTAL, HEAD_DIM, dtype=fd) for _ in range(N_LAYERS)],
                           [torch.zeros(1, KV_HEADS, MAX_TOTAL, HEAD_DIM, dtype=fd) for _ in range(N_LAYERS)])

    class EncodeWrapper(torch.nn.Module):
        """SigLIP x3 + prefix embed + PREFILL the fixed-max cache -> first_logits + 36 cache tensors."""
        def forward(self, img0, img1, img2, lang_tokens):
            b = lang_tokens.shape[0]
            bos = torch.full((b, 1), BOS, dtype=lang_tokens.dtype, device=lang_tokens.device)
            toks = torch.cat([lang_tokens, bos], dim=1)
            pe = torch.cat([*[pwe.embed_image(x) for x in (img0, img1, img2)],
                            pwe.embed_language_tokens(toks) * math.sqrt(WIDTH)], dim=1)  # [1,PREFIX_LEN,2048]
            att = torch.zeros(1, 1, PREFIX_LEN, MAX_TOTAL, dtype=pe.dtype)
            att[:, :, :, PREFIX_LEN:] = OPENPI_ATTENTION_MASK_VALUE       # prefix bidir over 0:PREFIX_LEN
            pos = torch.arange(PREFIX_LEN)[None]
            cache_position = torch.arange(PREFIX_LEN)
            (hidden, _), pkv = pwe.forward(attention_mask=att, position_ids=pos, past_key_values=_zero_cache(),
                                           inputs_embeds=[pe, None], use_cache=True, adarms_cond=[None, None],
                                           cache_position=cache_position)
            logits = lm_head(hidden[:, -1:, :])[:, 0]
            return (logits.to(fd), *pkv.flatten())

    class DecodeWrapper(torch.nn.Module):
        """ONE token: write its K/V at cache_position -> next logits + updated cache. O(1)/step."""
        def forward(self, new_token, att_4d, position_id, cache_position, *cache_tensors):
            cache = TensorCache(cache_tensors[:N_LAYERS], cache_tensors[N_LAYERS:])
            emb = pwe.embed_language_tokens(new_token) * math.sqrt(WIDTH)
            (hidden, _), pkv = pwe.forward(attention_mask=att_4d, position_ids=position_id,
                                           past_key_values=cache, inputs_embeds=[emb.to(fd), None],
                                           use_cache=True, adarms_cond=[None, None], cache_position=cache_position)
            logits = lm_head(hidden[:, -1:, :])[:, 0]
            return (logits.to(fd), *pkv.flatten())

    d = dict(
        img=torch.zeros(1, 3, 224, 224, dtype=fd), tok=torch.zeros(1, TOK, dtype=torch.long),
        new_token=torch.zeros(1, 1, dtype=torch.long),
        att_4d=torch.zeros(1, 1, 1, MAX_TOTAL, dtype=fd), position_id=torch.zeros(1, 1, dtype=torch.long),
        cache_position=torch.zeros(1, dtype=torch.long),
        cache=[torch.zeros(1, KV_HEADS, MAX_TOTAL, HEAD_DIM, dtype=fd) for _ in range(2 * N_LAYERS)],
    )
    return EncodeWrapper().eval(), DecodeWrapper().eval(), d


def _disk_free_gb() -> float:
    import shutil
    return shutil.disk_usage("/").free / 1e9


def cmd_export(out: Path, fp16: bool = True, free_weights: bool = False):
    import torch
    enc, dec, d = _build_wrappers(fp16=fp16)
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    ex = [natural_image(g).to(d["img"].dtype) for _ in range(3)]      # distinct-image trace (degenerate-trace fix)
    enc_ep = torch.export.export(
        enc, args=(ex[0], ex[1], ex[2], d["tok"]), strict=False)
    torch.export.save(enc_ep, str(out / "encode.pt2"))
    del enc_ep
    print(f"ok: wrote {out}/encode.pt2 (prefix_len={PREFIX_LEN}) — {_disk_free_gb():.1f}GB free")
    dec_ep = torch.export.export(
        dec, args=(d["new_token"], d["att_4d"], d["position_id"], d["cache_position"], *d["cache"]),
        strict=False)
    torch.export.save(dec_ep, str(out / "decode_step.pt2"))
    print(f"ok: wrote {out}/decode_step.pt2 — {_disk_free_gb():.1f}GB free")
    print("next (venv-B): .venv/bin/python models/pi0fast/export.py --lower --out", out)


def cmd_lower(out: Path):
    # The AR decode cache is a FIXED-max [1,KV_HEADS,MAX_TOTAL,HEAD_DIM] TensorCache (write-at-
    # cache_position via functional index_copy) — static shapes, lowers cleanly. See _build_wrappers.
    import torch
    from coreai_torch import TorchConverter, get_decomp_table

    def _load(name):
        return torch.export.load(str(out / f"{name}.pt2")).run_decompositions(get_decomp_table())

    conv = TorchConverter()
    conv.add_exported_program(_load("encode"), input_names=_ENC_INPUTS, output_names=_ENC_OUTPUTS, entrypoint_name="encode")
    conv.add_exported_program(_load("decode_step"), input_names=_DEN_INPUTS, output_names=_DEN_OUTPUTS, entrypoint_name="decode_step")
    prog = conv.to_coreai()
    for name in ("encode", "decode_step"):
        p = out / f"{name}.pt2"
        if p.exists():
            p.unlink()
    prog.optimize()
    prog.save_asset(out / f"{out.name}.aimodel")
    print(f"ok: lowered pi0fast -> {out}/{out.name}.aimodel (encode, decode_step) — {_disk_free_gb():.1f}GB free")


def main():
    ap = argparse.ArgumentParser(description="pi0fast split-export")
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
