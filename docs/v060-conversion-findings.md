# LeRobot v0.6.0 → Apple Core AI — Conversion Findings

**Program:** convert every LeRobot v0.6.0 policy to an on-device Core AI `.aimodel`
via `coreai-fabric`, verify a Gate B fidelity number, and register it in
`coreai-catalog`. Runtime: `coreai_torch 0.4.1` / `coremltools 9.0`, target
macOS/iOS 27 / Apple Silicon. Dated 2026-07-07.

Visual report: an Artifact of this document is published for sharing; this file is
the durable, versioned reference.

---

## The one distinction that predicts everything

A v0.6.0 policy is easy or hard on-device for exactly one reason — whether its
**deployable graph is a small separable head** or the **entire multi-billion-param
backbone**.

- **Class A — separable head.** A distinct small action/reward module hangs off a
  standard VLM. Fabric exports just that module (tens of MB – ~1 GB); it lowers,
  loads, and verifies easily. Host owns the backbone.
  Members: `evo1` (flow-matching action head), `robometer` (MLP reward/progress
  heads), `groot` (DiT action head ~1B), `vla-jepa` (action head + qwen_context).
- **Class B — whole-model VLA.** Actions are denoised *inside* the transformer
  (interleaved tokens / per-layer conditioning / streaming caches); there is no
  small head to peel off. The deployable graph carries billions of params, so the
  fp16 asset (~10 GB) overruns disk and the ANE loader.
  Members: `lingbot_va` (Wan dual-stream DiT), `fastwam` (Wan action DiT), `eo-1`
  (denoise inside Qwen2.5-VL), `molmoact2` (per-layer-KV action expert).

The split discipline is constant: **host keeps the backbone, fabric ships the
deployable graph + a Gate B number** that certifies the export matches the torch
reference (not downstream task success).

---

## Findings — the discovery chain

**F1 · An MPSGraph pass segfaults on mask-free attention.**
EVO1's `.aimodel` crashed on load in MetalPerformanceShadersGraph's
`FoldMultiplyIntoSDPAScale` (it folds a preceding LayerNorm gamma into a
reconstructed SDPA scale and dies). Ablation: the crash needs an **unmasked**
cross-attention. **Fix:** inject an all-false `key_padding_mask` so the masked-SDPA
path is taken. It must be **data-dependent** (`sum(|ctx|) < -1`) — a constant
all-false mask is folded away and re-crashes. (macOS 27 / Xcode-27 beta.)

**F2 · coreai_torch has no complex dtype.**
Wan RoPE uses `torch.polar` / `view_as_complex` → `KeyError: torch.complex128`.
**Fix:** rewrite the rotation in reals from the 3D frequency bands —
`(xe+i·xo)(cos+i·sin) = (xe·cos−xo·sin, xe·sin+xo·cos)`; pass `cos`/`sin` as real
tensors. Verified numerically identical, **max abs diff 1.7e-6**.

**F3 · The `flex_attention` scare was a red herring.**
LingBot-VA seems to need `flex_attention` (unlowerable), but its config sets
`attn_mode: torch` → inference runs plain `F.scaled_dot_product_attention`
(`custom_sdpa`); flex is training-only. **No attention rewrite needed.**

**F4 · Restricted licenses are index-only, by policy.**
GR00T converts cleanly, but `coreai-fabric publish` **refuses** to republish
converted weights of a restricted (NVIDIA Open Model License, non-commercial)
upstream — no override flag. Correct deliverable: the **reproducible recipe** +
upstream pointer. Conversion done; redistribution intentionally not.

**F5 · int8 breaks the Class-B disk / ANE wall. (the unlock)**
LingBot-VA's action graph is the full 5B Wan DiT. Its fp16 asset (**10 GB**) built
but failed to load — ANE `Program load failure 0x10004` + disk exhaustion. Applying
**torchao `Int8WeightOnlyConfig` before `torch.export`** halves it to **~5 GB**
(≈1 byte/param), which saves, **loads**, and verifies at
`graph_output_cosine 0.9999999999990`. This put a 5B video-world-model on-device and
generalizes to every remaining Class-B policy.
> torchao 0.17 API is `Int8WeightOnlyConfig` (not the old `int8_weight_only`).

**Supporting trick (tight disk):** a combined export+parity driver loads the
weights, **deletes the multi-GB source file** while the model stays resident in RAM,
then writes the asset into the freed space and runs parity from the in-RAM model.

---

## Gate B ledger

| Model | Deployable graph | Metric | Worst value | Precision | Status |
|---|---|---|---|---|---|
| `evo1-so100` | flow-matching action head | action_parity | 0.9999999999999983 | fp32 | registered |
| `robometer-4b` | reward + progress heads | graph_output_cosine | 0.999999999996286 | fp32 | registered |
| `groot-n1-7-3b` | Cosmos + DiT action head | action_parity | 0.9999484569893827 | fp32 | index-only |
| `lingbot-va-base` | Wan 5B action denoiser | graph_output_cosine | 0.9999999999990612 | int8 | published |
| `vla-jepa` (×3) | action head + qwen_context | action_parity / cosine | ≈1.0 / 0.9997 | fp32 | registered |

Gate A checks bundle structure; Gate B is conversion fidelity vs. the torch
reference on seeded inputs.

---

## Status

- **Published + registered:** VLA-JEPA ×3, EVO1, Robometer.
- **Published + register PR open (#30):** LingBot-VA (int8) — first 5B
  video-world-model on Core AI.
- **Converted + verified, index-only:** GR00T (non-commercial license).
- **int8 route validated, per-model build queued:** FastWAM, EO-1, MolmoAct2.
- **Discarded:** TOPReward (zero-shot wrapper, no weights of its own).
- **Catalog:** 124 models (consolidated PR #29 merged) + 4 community zoo models
  indexed and a new `diarization` bundle_kind.

---

## Reusable playbook

1. **Split, don't swallow** — host owns the VLM; fabric ships a fixed-shape,
   stateless task head; Gate B compares torch-ref vs. asset on seeded inputs.
2. **Real RoPE, always** — never let `view_as_complex` reach the converter; emit
   `cos`/`sin` and rotate in reals.
3. **Mask the SDPA** — a data-dependent all-false `key_padding_mask` dodges the
   MPSGraph fold-into-scale segfault (macOS-27 beta).
4. **int8 for whole-model VLAs** — `quantize_(m, Int8WeightOnlyConfig())` before
   `torch.export`; ~1 byte/param fits disk + ANE, near-lossless export fidelity.
5. **Delete-after-load** — on tight disk, free the source safetensors after loading
   into RAM, before `save_asset`; combine export + parity in one process.
6. **Read the license before the code** — restricted upstream → index-only: ship
   the reproducible recipe, never the reweighted binary. Fabric enforces this.

---

*Lanes: `models/{evo1,robometer,groot,lingbot_va}/`. Recipes: `recipes/*.yaml`.
Every parity number is reproducible from a committed recipe. Not affiliated with Apple.*
