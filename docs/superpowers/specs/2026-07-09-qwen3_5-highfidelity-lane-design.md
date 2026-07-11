# Qwen3.5 high-fidelity Core AI lane — design

**Status:** Design (brainstormed 2026-07-09). Awaiting review → implementation plan.
**Pilot model:** `empero-ai/Qwythos-9B-Claude-Mythos-5-1M` (rev `763f72fc2c3b186e977adcbaac0c18128f182166`, apache-2.0, 9.41 B params).
**Reference impl:** `transformers` `Qwen3_5*` (`.venv-lerobot`, transformers 5.3.0; config declares 5.12.1) — `modeling_qwen3_5.py`.

## 1. Goal & scope

Build a **reusable `qwen3_5` architecture lane** in coreai-fabric that exports a Qwen3.5 VLM to a Core AI `.aimodel` bundle at **high fidelity** — no downgrades of the model's own math. Qwythos is the first client; the lane must generalize to any dense `qwen3_5` / `qwen3_5_vision` checkpoint.

This lane is also the **pilot exemplar** for a new ecosystem mandate: agentic, iterative optimization with **full provenance** — every decision (gain *or* loss, where and how) recorded so results are replicable and independently validatable. See §10.

**In scope:** dense `qwen3_5` text decoder (hybrid linear+full attention), `qwen3_5_vision` tower (dynamic resolution), M-RoPE, YaRN-configured rope, 128 K validated context, the experiment ledger, parity harness, recipe/register/publish integration.

**Non-goals (v1):**
- **MTP** (`mtp_num_hidden_layers: 1`) — the multi-token-prediction head is dropped; export the base decoder. (Revisit as a speculative-decode optimization, ledgered.)
- **256 K / 1 M context** — 256 K is the native ceiling, 1 M is YaRN-extended; both are follow-on optimizations gated on KV-cache compression (§4).
- **`qwen3_5_moe`** — a sibling arch; out of scope.
- The **ecosystem-wide** provenance standard — this lane pilots it; the general rollout is its own spec (§14).

## 2. Fidelity policy

High fidelity = replicate the model's own math exactly; the only knobs we bound are deployment envelope (context length, quantization), never the architecture.

- **Gated linear attention (DeltaNet):** exact — correctness, not a knob.
- **M-RoPE:** preserved (`mrope_interleaved: true`, `mrope_section: [11,11,10]`), *not* downgraded to 1-D RoPE (the Apple Qwen3-VL backbone's downgrade is exactly what we reject).
- **YaRN rope:** replicated (`rope_type: yarn`, `factor: 4.0`, `original_max_position_embeddings: 262144`, `rope_theta: 1e7`). YaRN frequency scaling is baked into the shipped config, so it is applied even below 256 K; we replicate it exactly.
- **Vision:** native **dynamic resolution** (NaViT-style, pixel-count bounds), not fixed 448 px.
- Quantization is a *bounded, ledgered* optimization (§10), not a fidelity compromise: every quant choice records its measured parity/quality delta.

## 3. Architecture reality (grounded)

Dense hybrid transformer. `text_config`: hidden 4096, 32 layers, `head_dim` 256, GQA 16 Q / 4 KV, intermediate 12288, vocab 248320, `rms_norm_eps` 1e-6, SiLU, `tie_word_embeddings: false`. `image_token_id` 248056, `vision_start_token_id` 248053, eos `[248046, 248044]`, pad 248044.

`layer_types` = 8 × `[linear, linear, linear, full]` → **24 GatedDeltaNet + 8 full-attention** layers (`full_attention_interval: 4`).

**Memory consequence (the reason high context is affordable):** only the 8 full-attention layers carry a growing KV cache; the 24 linear layers carry **fixed-size** state (§7). Long-context cost is ~4× lower than a dense transformer.

## 4. Context strategy

- **v1 headline (validated + published): 128 K.** Sits within the native 256 K window (`original_max_position_embeddings: 262144`), so no extrapolation stress; the export is parameterized (`--max-context-length`).
- **256 K** = native ceiling — follow-on; needs KV-cache budgeting for the 8 full layers.
- **1 M** = YaRN-extended (`factor 4.0`) — follow-on; needs KV-cache compression/quantization (a distinct subsystem, ledgered).
- KV cache lives in **8/32** layers only; the 24 linear layers are O(1) regardless of length.

## 5. Module decomposition

New package `models/qwen3_5/`. Each unit has one purpose and a typed interface, testable in isolation.

| # | Module | Purpose | Interface |
|---|--------|---------|-----------|
| 1 | `spec.py` | `config.json` → typed `Qwen35Spec` (layer_types, dims, rope/yarn, vision geometry, token ids). Single source of truth. | `load_spec(repo, rev) -> Qwen35Spec` |
| 2 | `weights.py` | safetensors → named tensors by prefix (`model.language_model.*`, `model.visual.*`). | `load_weights(spec) -> WeightBook` |
| 3 | `linear_attn.py` | **Core 1.** GatedDeltaNet layer → stateful recurrent Core AI graph. | `build_linear_layer(spec, wb, i) -> Graph` |
| 4 | `full_attn.py` | **Core 2.** Softmax GQA + partial M-RoPE + YaRN + KV cache. | `build_full_layer(spec, wb, i) -> Graph` |
| 5 | `vision.py` | `qwen3_5_vision` dynamic-res ViT → Core AI graph (bespoke, or borrow Apple's if the spike proves byte-compat). | `build_vision(spec, wb) -> Graph` |
| 6 | `export.py` + `run_export.py` | Assemble embed + vision + 24 linear + 8 full → `.aimodel` bundle + metadata + host orchestration. CLI entry. | `export(spec) -> BundlePath` |
| 7 | `parity.py` | Two-venv greedy parity A/B/C, extended for recurrent state + M-RoPE, up to 128 K. | Gate B ≥ 0.999 |
| 8 | recipe + `register` + `publish` | Recipe YAML, `coreai-fabric register` → catalog PR, publish gate (apache-2.0 passes). | catalog artifact |
| 9 | `ledger.py` + `provenance/*.jsonl` | **Cross-cutting.** Experiment ledger (§10). | `record(entry)` |

## 6. Core 1 — GatedDeltaNet → recurrent Core AI graph

Source: `modeling_qwen3_5.py:446-627` (`Qwen3_5GatedDeltaNet`).

**Dims (Qwythos):** `linear_key_head_dim` 128 × `linear_num_key_heads` 16 → key_dim 2048; `linear_value_head_dim` 128 × `linear_num_value_heads` 32 → value_dim 4096; `conv_dim` = key_dim·2 + value_dim = **8192**; `linear_conv_kernel_dim` 4. `num_v_heads // num_k_heads = 2` → q,k `repeat_interleave(2)`.

**Decode step (seq_len = 1) — the on-device path:**
1. Projections: `qkv = in_proj_qkv(h)`; `z = in_proj_z(h)`; `b = in_proj_b(h)`; `a = in_proj_a(h)`.
2. **State 1 — conv_state** `[conv_dim=8192, kernel-1=3]`: `causal_conv1d_update` (depthwise, kernel 4) then SiLU; roll one step.
3. Split → q, k, v; L2-norm q, k.
4. Gates: `β = sigmoid(b)`; `g = −exp(A_log) · softplus(a + dt_bias)` **in fp32**.
5. **State 2 — recurrent_state S** `[num_v_heads=32, head_k=128, head_v=128]` (524 288 elems/layer): delta rule
   `S_t = diag(decay(g))·S_{t-1} + β·(v − S_{t-1}k)kᵀ`; `out = q·S_t`.
6. `RMSNormGated(out, z)` → `out_proj` (4096→4096) → h′.

**Prefill:** reference uses `chunk_gated_delta_rule` (chunked parallel scan); decode uses `recurrent_gated_delta_rule`. We lower **both to plain Core AI graph ops** (no `fla`/`causal-conv1d` kernels). Parity must show chunk-prefill and step-recurrent agree (§11, spike S1).

**Two carried states per linear layer** (conv_state + recurrent_state) are the KV-cache analog but **fixed size** → export them as graph state I/O, following the RWKV7 recurrent-decode precedent (zoo) and fabric's StaticCache (pi0fast).

## 7. Core 2 — full attention

Source: `modeling_qwen3_5.py:714` (`Qwen3_5Attention`), `apply_rotary_pos_emb:638`.

- GQA 16 Q / 4 KV, `head_dim` 256.
- **Partial rotary:** `partial_rotary_factor: 0.25` → rotary_dim = 64; only the first 64 dims rotate, `q_pass`/`k_pass` (192 dims) pass through unrotated.
- **M-RoPE:** `mrope_interleaved: true`, `mrope_section [11,11,10]` (sums to 32 = rotary_dim/2) — 3 positional axes (temporal/height/width) for the VLM.
- **YaRN:** applied to the rope frequencies (`factor 4.0`, `original_max_position 262144`, `theta 1e7`).
- **KV cache** — the only growing state; 8 layers.

## 8. Vision tower

Source: `Qwen3_5VisionModel:1088`. `qwen3_5_vision`: depth 27, hidden 1152, 16 heads, patch 16, `spatial_merge_size` 2, `temporal_patch_size` 2, `out_hidden_size` 4096, `deepstack_visual_indexes: []`, `num_position_embeddings` 2304. Preprocessor: mean/std 0.5, **dynamic resolution by pixel count** (shortest 65536 / longest 16 777 216 px²).

**Spike S3 decides** bespoke vs borrow: if `qwen3_5_vision` graph/weights are byte-compatible with Apple's exporter, borrow the encoder as an optimization; else build bespoke (dynamic-res ViT + `Qwen3_5VisionPatchMerger`). High-fidelity (dynamic res) likely forces bespoke — the spike measures it, ledgered.

## 9. Bundle & host orchestration

`.aimodel`/`.vlm` bundle: **vision** (pixel_values → image_features) + **embed** (input_ids → embeddings) + **main** (inputs_embeds + position_ids → logits, stateful: conv_states×24, recurrent_states×24, k/v_cache×8). Host: vision-encode → text-embed → splice image features at `image_token_id` (248056) → prefill → greedy decode. Metadata records layer_types, state shapes, rope/yarn params, vision geometry.

## 10. Provenance / experiment ledger (the pilot discipline)

Every optimization attempt (quantization, KV compression, YaRN extension, vision-res, kernel-free lowering variants) is an **immutable, committed, machine-readable** record in `models/qwen3_5/provenance/*.jsonl`, linked from the model card and the catalog artifact provenance. **Negative results are recorded too.**

```json
{
  "id": "exp-0007",
  "hypothesis": "int8 per-block32 on linear out_proj holds parity",
  "target": {"model": "qwythos-9b", "weights_rev": "763f72f...", "component": "linear_attn.out_proj"},
  "config_hash": "sha256:…", "seed": 0, "env_fingerprint": {"os":"…","py":"…","transformers":"5.3.0","coreai":"…"},
  "deltas": {"greedy_parity": 0.998, "first_tok_cos": 0.9997, "latency_ms_tok": 41.2, "peak_mem_gb": 9.8, "quality": {"mmlu_delta": -0.1}},
  "verdict": "kept", "why": "parity ≥0.999 gate met at −0.2% mem",
  "repro_cmd": ".venv/bin/python models/qwen3_5/parity.py --compare --exp exp-0007 …"
}
```

Fabric's existing publish provenance and privacy guard (`publish.py`) consume/gate on it. No hardware fingerprints leak (existing guard).

## 11. Parity & verification

Two-venv greedy parity (reference `.venv-lerobot` transformers → `.npz`; `--compare` drives the Core AI asset), extended from `models/vlm/parity.py`:

- **A — greedy token-exact** over `decode_len` (the "X/Y vs fp32" number) — PRIMARY.
- **B — first-token logit cosine + argmax** — isolates vision+embed+prefill from decode drift. **Gate B ≥ 0.999.**
- **C — vision-feature cosine** (coreai vs torch) — self-validates image preprocessing.
- **D (new) — recurrent-state fidelity:** conv_state + recurrent_state cosine vs torch at N steps — isolates the DeltaNet lowering.

Gate A (structural: bundle files, metadata parses/matches recipe) + Gate B (numeric ≥0.999) as in existing recipes.

## 12. De-risk spike order (empirical; each spike writes ledger entries)

1. **S1 — DeltaNet decode graph** (highest risk): one linear layer, kernel-free, decode-step; prove recurrent-state cosine + chunk-vs-recurrent agreement on random + real hidden. Gate D.
2. **S2 — full-attn partial M-RoPE + YaRN**: one full layer; first-token logit cosine vs torch.
3. **S3 — vision compat**: borrow-vs-bespoke measurement for `qwen3_5_vision` dynamic-res.
4. **S4 — full-stack 128 K greedy parity** on Qwythos; Gate B.
5. **S5 — quantization sweep** (int8/int4 presets) under the ledger; pick the parity-preserving preset.

Stop-on-red: a failing spike halts and records *why* before proceeding.

## 13. Fabric integration

- Recipe `recipes/qwen3_5-qwythos-9b.yaml`: `tool: models/qwen3_5/run_export.py` (script-tool, off Apple's `SUPPORTED_MODELS`), `parity`, `publish` (namespace `kevinqz`), `catalog` (bundle_kind `vlm`, min_os 27.0).
- `coreai-fabric register` → catalog PR (needs the catalog to accept a `qwen3_5`/hybrid entry — coordinate with catalog schema, may need a capability/bundle note).
- **Publish gate:** apache-2.0 is on the permissive allowlist → passes; ship upstream LICENSE/NOTICE, clean bundle (no raw weights), no local paths (existing guards).

## 14. Follow-on (separate specs)

1. **Ecosystem-wide provenance standard** — generalize §10 across catalog + fabric + zoo (the mandate).
2. **256 K / 1 M context** — KV-cache compression for the 8 full layers.
3. **MTP speculative decode** — re-add the dropped MTP head as a latency optimization.
4. **qwen3_5_moe** lane.

## 15. Open questions

- Does Apple's Core AI runtime express the DeltaNet recurrence (outer products + per-head decay) as graph state I/O without a custom kernel? (S1 answers.)
- Chunk-prefill parity: can we prefill via the recurrent form, or must we lower the chunked scan? (S1.)
- Catalog schema: does a hybrid linear+full VLM need a new `bundle_kind` note or capability, or does `vlm` suffice?
