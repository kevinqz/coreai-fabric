"""Inject fabric's extra VLMSpec entries into coreai_models.vlm.export.SUPPORTED_MODELS at runtime.

coreai.vlm.export gates on a hardcoded SUPPORTED_MODELS dict keyed by short-name (only 'qwen3-vl'
-> Qwen/Qwen3-VL-2B-Instruct); there is NO raw-HF-id / --experimental path. The backbone
(Qwen3VLForCausalLMEmbeddings) is arch-generic — it reads text_config dynamically and loads weights
by the model.language_model.* / model.visual.* prefixes — so ANY genuine qwen3_vl checkpoint exports
once its spec (HF id + vision geometry) is registered. This module adds those specs WITHOUT patching
Apple's site-packages (which a .venv rebuild would clobber).

Usage: import this module (it self-registers on import) BEFORE calling the export, e.g.
    python -c "import models.vlm.register_specs; from coreai_models.vlm.export import main; main()" gelab-zero-4b ...
"""
from coreai_models.vlm.export import SUPPORTED_MODELS, VLMSpec

# microsoft/GELab-Zero-4B-preview-Sico-Evolution — genuine qwen3_vl (Qwen3VLForConditionalGeneration),
# text hidden 2560 / 36 layers / vocab 151936, tie_word_embeddings True (backbone reads it dynamically).
# Vision geometry from its config.json + preprocessor_config.json: patch 16, merge 2, temporal 2,
# image_token_id 151655 (same as the 2B), image_mean/std [0.5,0.5,0.5] (DIFFERS from the 2B's CLIP norm).
# image_size fixed to 448 (the vision graph is single-resolution; 448/16/2 -> 196 visual tokens).
_EXTRA = {
    "gelab-zero-4b": dict(
        short_name="gelab-zero-4b",
        hf_model_id="microsoft/GELab-Zero-4B-preview-Sico-Evolution",
        output_name="gelab_zero_4b",
        image_token_id=151655,
        image_size=448,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        rescale_factor=1.0,   # matches the qwen3-vl 2B spec convention; confirm vs observed 2B metadata
    ),
}

for name, kw in _EXTRA.items():
    if name not in SUPPORTED_MODELS:
        SUPPORTED_MODELS[name] = VLMSpec(**kw)


# ---- tokenizer-save fix ----
# coreai.vlm.export does AutoTokenizer.from_pretrained(model_dir) with no use_fast; for Qwen3-VL that
# resolves to the SLOW Qwen2Tokenizer whose vocab_file is None -> TypeError, crashing the export AFTER
# main+embed but BEFORE vision + tokenizer/ + the VLM metadata.json. Force the fast tokenizer (the repo
# ships tokenizer.json + vocab.json + merges.txt; Qwen2TokenizerFast loads + saves fine).
import coreai_models.vlm.export as _vx  # noqa: E402

_orig_auto_from_pretrained = _vx.AutoTokenizer.from_pretrained


def _fast_from_pretrained(cls_or_path, *a, **k):
    k.setdefault("use_fast", True)
    return _orig_auto_from_pretrained(cls_or_path, *a, **k)


_vx.AutoTokenizer.from_pretrained = _fast_from_pretrained
