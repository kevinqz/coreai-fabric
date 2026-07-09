# Upstream coverage — robbyant / feyninc / sensenova

Gap analysis of the requested upstream collections vs. the fabric recipe set.
Snapshot: **2026-07-08**. Legend: ✅ have · ⬜ missing.

## Summary

| Upstream | Members | Covered | Missing |
|---|---|---|---|
| github robbyant/lingbot-vla-v2 | 1 | 1 | 0 |
| hf robbyant/lingbot-vision | 4 | 4 | 0 |
| hf feyninc/pulpie | 3 | 2 | **1** |
| hf sensenova/SenseNova-Vision-7B-MoT | 1 | 1 | 0 |
| hf robbyant/lingbot-video | 3 | 0 | **3** |
| hf robbyant/lingbot-world-v2 *(bonus, done this session)* | 2 | 1 | 0 |

**Net: 4 missing** — `pulpie-orange-small` + the entire `lingbot-video` collection (3).

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

## 3. feyninc/pulpie — ⬜ 2/3 (1 missing)

Token-classification models ("cleaning up the web").

| Model | Params | Recipe | Status |
|---|---|---|---|
| ⬜ **pulpie-orange-small** | 0.2B | — | **MISSING** |
| ✅ pulpie-orange-base | 0.6B | `recipes/pulpie-orange-base.yaml` | registered |
| ✅ pulpie-orange-large | 2B | `recipes/pulpie-orange-large.yaml` | registered |

**Action:** add `pulpie-orange-small` — same encoder lane as base/large, smallest
of the family, should be a near-copy of the base recipe. Low risk, quick win.

## 4. sensenova/SenseNova-Vision-7B-MoT — ✅ HAVE (draft)

| Model | Recipe | Status |
|---|---|---|
| ✅ SenseNova-Vision-7B-MoT | `recipes/sensenova-vision-7b-mot.yaml` | **draft** (index-only) |

Index-only draft (CC-BY-NC-4.0). Deployable core = the SigLIP ViT encoder; the
`export.py` driver is not yet written. Blocked from publishing on license, not effort.

## 5. robbyant/lingbot-video — ⬜ 0/3 (all missing)

**Distinct from `lingbot-world-v2`.** A separate video-model collection, entirely
uncovered.

| Model | Notes | Recipe | Status |
|---|---|---|---|
| ⬜ **lingbot-video-dense-1.3b** | 1.3B dense video model — smallest, best first target | — | **MISSING** |
| ⬜ **lingbot-video-moe-30b-a3b** | 30B MoE, ~3B active — large, MoE graph-split territory | — | **MISSING** |
| ⬜ **lingbot-video-rewriter-lora** | prompt-rewriter LoRA — likely a text adapter, not a core convert target | — | **MISSING** |

**Action:** triage needed — architecture unknown (probably Wan-family like world-v2).
Best entry = `dense-1.3b` (smallest). The `moe-30b-a3b` is a big MoE (graph-split, cf.
lingbot-vla-v2). The `rewriter-lora` is likely out of scope for a `.aimodel` lane.

## 6. robbyant/lingbot-world-v2 — ✅ done this session (bonus)

| Model | Recipe | Status |
|---|---|---|
| ✅ lingbot-world-v2-14b-causal-fast-diffusers | `recipes/lingbot-world-v2-14b-causal-fast.yaml` | **verified** (index-only) |
| n/a lingbot-world-v2-14b-causal-fast (native) | same weights, -diffusers is the conversion base | — |

First generative-video lane. VAE decoder built + Gate B min 0.99999999999941 (n=8),
fp32 single-program `.aimodel`. Index-only (CC-BY-NC-SA). Follow-ups: streaming
feat_cache-as-I/O, then the 14B DiT graph-split. See `docs/validation-log.md`.

## Prioritized backlog

1. **pulpie-orange-small** (0.2B) — trivial, completes the pulpie family.
2. **lingbot-video-dense-1.3b** — smallest video model; triage architecture first.
3. **SenseNova export.py** — turn the draft into a measured Gate-B number (index-only).
4. **lingbot-world-v2 streaming** — feat_cache-as-I/O, finishes the VAE lane.
5. **lingbot-video-moe-30b-a3b** — big MoE graph-split; dedicated campaign.
6. **lingbot-world-v2 DiT 14B** — the SotA core; multi-session (~28GB download + split).
