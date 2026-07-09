# Upstream coverage — robbyant / feyninc / sensenova

Gap analysis of the requested upstream collections vs. the fabric recipe set.
Snapshot: **2026-07-08** (updated after the backlog sweep). Legend: ✅ have ·
⬜ missing · 🚫 out-of-scope.

## Summary

| Upstream | Members | Covered | Missing |
|---|---|---|---|
| github robbyant/lingbot-vla-v2 | 1 | 1 | 0 |
| hf robbyant/lingbot-vision | 4 | 4 | 0 |
| hf feyninc/pulpie | 3 | **3** | 0 |
| hf sensenova/SenseNova-Vision-7B-MoT | 1 | 1 | 0 |
| hf robbyant/lingbot-video | 3 | **2** + 1🚫 | 0 |
| hf robbyant/lingbot-world-v2 | 2 | 1 | 0 |

**All requested models now have a recipe.** The backlog sweep added
`pulpie-orange-small` (verified), `lingbot-video-dense-1.3b` (verified, publishable),
`lingbot-video-moe-30b-a3b` (draft — VAE verified by identity); the
`lingbot-video-rewriter-lora` is out-of-scope (prompt LoRA, not a `.aimodel` lane).

**Family VAE fact:** world-v2, video-dense-1.3b and video-moe-30b all ship the
**same** AutoencoderKLWan (vae/ safetensors byte-identical, SHA256 d6e524b3…) — ONE
verified VAE-decoder asset covers all three.

## 1. github.com/Robbyant/lingbot-vla-v2 — ✅ HAVE

The code repo for `robbyant/lingbot-vla-v2-6b`.

| Model | Recipe | Status |
|---|---|---|
| ✅ robbyant/lingbot-vla-v2-6b | `recipes/lingbot-vla-v2.yaml` | **registered** → kevinqz/LingBot-VLA-2.0-CoreAI |

First Mixture-of-Experts VLA on the ANE (36-layer MoE, 4-program graph-split, fp16,
Gate B min 0.99948). Nothing to do.

## 2. robbyant/lingbot-vision — ✅ COMPLETE (4/4)

| Model | Recipe | Status |
|---|---|---|
| ✅ vit-small | `recipes/lingbot-vision-vit-small.yaml` | registered |
| ✅ vit-base | `recipes/lingbot-vision-vit-base.yaml` | registered |
| ✅ vit-large | `recipes/lingbot-vision-vit-large.yaml` | registered |
| ✅ vit-giant | `recipes/lingbot-vision-vit-giant.yaml` | registered |

The `graph_output_cosine` encoder lane. All four published+registered. Nothing to do.

## 3. feyninc/pulpie — ✅ COMPLETE (3/3)

Token-classification models ("cleaning up the web").

| Model | Params | Recipe | Status |
|---|---|---|---|
| ✅ pulpie-orange-small | 0.2B | `recipes/pulpie-orange-small.yaml` | **verified** (index-only, cc-by-nc) |
| ✅ pulpie-orange-base | 0.6B | `recipes/pulpie-orange-base.yaml` | registered |
| ✅ pulpie-orange-large | 2B | `recipes/pulpie-orange-large.yaml` | registered |

`pulpie-orange-small` built this session: EuroBERT-210m, Gate B min 0.99999999999174
(n=8). Note the small checkpoint is **cc-by-nc-4.0** (base/large are apache-2.0), so
it's index-only — verified but not republished.

## 4. sensenova/SenseNova-Vision-7B-MoT — ✅ HAVE (draft)

| Model | Recipe | Status |
|---|---|---|
| ✅ SenseNova-Vision-7B-MoT | `recipes/sensenova-vision-7b-mot.yaml` | **draft** (index-only) |

Index-only draft (CC-BY-NC-4.0). Deployable core = the SigLIP ViT encoder; the
`export.py` driver is not yet written. Blocked from publishing on license, not effort.

## 5. robbyant/lingbot-video — ✅ covered (2 verified/draft + 1 out-of-scope)

**Distinct from `lingbot-world-v2`** but **Apache-2.0 (publishable)** and shares the
same VAE. `LingBotVideoPipeline` (T2I/T2V/TI2V).

| Model | Notes | Recipe | Status |
|---|---|---|---|
| ✅ lingbot-video-dense-1.3b | VAE decoder, Gate B 0.99999999999941 | `recipes/lingbot-video-dense-1.3b.yaml` | **published** ([HF](https://huggingface.co/kevinqz/LingBot-Video-Dense-1.3B-CoreAI), [PR #36](https://github.com/kevinqz/coreai-catalog/pull/36)) |
| ✅ lingbot-video-moe-30b-a3b | VAE decoder built from its own (identical) weights, Gate B 0.99999999999941 | `recipes/lingbot-video-moe-30b-a3b.yaml` | **published** ([HF](https://huggingface.co/kevinqz/LingBot-Video-MoE-30B-A3B-CoreAI), [PR #37](https://github.com/kevinqz/coreai-catalog/pull/37)) |
| 🚫 lingbot-video-rewriter-lora | prompt-rewriter LoRA — a text adapter, not a `.aimodel` core | — | out-of-scope |

Both VAE decoders are **published+registered** (Apache-2.0) — the first video assets
in the fleet. Both were built from their own vae/ safetensors, both byte-identical
(SHA256 d6e524b3…) to world-v2's VAE. Each repo's DiT is a dedicated campaign; the
rewriter LoRA is out of scope. Recipes stay `published` until PRs #36/#37 merge, then
`register --mark-merged` flips them to `registered`.

## 6. robbyant/lingbot-world-v2 — ✅ done this session (bonus)

| Model | Recipe | Status |
|---|---|---|
| ✅ lingbot-world-v2-14b-causal-fast-diffusers | `recipes/lingbot-world-v2-14b-causal-fast.yaml` | **verified** (index-only) |
| n/a lingbot-world-v2-14b-causal-fast (native) | same weights, -diffusers is the conversion base | — |

First generative-video lane. VAE decoder built + Gate B min 0.99999999999941 (n=8),
fp32 single-program `.aimodel`. Index-only (CC-BY-NC-SA). Follow-ups: streaming
feat_cache-as-I/O, then the 14B DiT graph-split. See `docs/validation-log.md`.

## Prioritized backlog

- ~~pulpie-orange-small~~ ✅ done (verified, index-only)
- ~~lingbot-video-dense-1.3b VAE~~ ✅ done (verified, publishable)

- ~~Publish lingbot-video-dense-1.3b~~ ✅ published+registered (HF + catalog PR #36)
- ~~Publish lingbot-video-moe-30b-a3b VAE~~ ✅ published+registered (HF + catalog PR #37)

Remaining:
1. **Merge catalog PRs #36/#37**, then `register --mark-merged` on both.
2. **SenseNova export.py** — turn the draft into a measured Gate-B number (index-only).
3. **Streaming feat_cache-as-I/O** — finishes the VAE lane for the whole video family
   (world-v2 + video-dense + video-moe share the asset).
4. **lingbot-video DiT** — dense-1.3b DiT first (smallest, Apache, publishable), then
   the 30B MoE graph-split (cf. lingbot-vla-v2), then world-v2's 14B DiT. Multi-session.
