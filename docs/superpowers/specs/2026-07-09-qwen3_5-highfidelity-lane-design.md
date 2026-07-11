# Qwen3.5 high-fidelity Core AI lane — design (v2, validated)

**Status:** Design, revised after a 4-agent validation pass (catalog / fabric / modeling / runtime) against the live code on 2026-07-09. Awaiting review → implementation plan.
**Pilot model:** `empero-ai/Qwythos-9B-Claude-Mythos-5-1M` (rev `763f72fc2c3b186e977adcbaac0c18128f182166`, apache-2.0, 9.41 B).
**Reference impl:** `transformers` `Qwen3_5*` — `modeling_qwen3_5.py` (verified runnable in `.venv-lerobot`, transformers 5.3.0, pure-torch fallback path).

> **v2 headline (validation).** The Qwen3.5 hybrid **text decoder is already solved and shipping** in the zoo: **Ornith-1.0-9B** is field-for-field the same arch as Qwythos and ships in `apps/CoreAIChatMac` (int8hu, ~10 GB, gated fp16/int8/int4 24/24 exact); **MiniCPM-V-4.6** is an already-shipped hybrid-GDN **VLM** reusing the zoo's `qwen3_5.py` overlay verbatim. So this lane is **mostly a port + the genuinely-new parts**, not a from-scratch build. New work = high-fidelity **dynamic-resolution vision**, **M-RoPE on the hybrid backbone**, **128 K prefill**, and the fabric **provenance/publish/catalog** plumbing.

## 1. Goal & scope

Bring a reusable, high-fidelity `qwen3_5` VLM export into coreai-fabric, with Qwythos as the first client and the **pilot exemplar** for the ecosystem's agentic-optimization + full-provenance mandate (§10). "High fidelity" = replicate the model's own math; the only bounded knobs are the deployment envelope (device, context, quant), each ledgered.

**Reuse (do not rebuild):** vendor/port the zoo's config-driven overlay `coreai-model-zoo/conversion/overlay/.../models/macos/qwen3_5.py` (+ `qwen3_5_config.py`, `qwen3_5_gdn_metal.py`, export scripts, `_smoke/test_ornith9b_eager_gate.py`). It already lowers the full GatedDeltaNet + full-attention hybrid at 9 B and drives a hybrid VLM. Fabric's *new* code is scoped to the four items above.

**In scope:** dense `qwen3_5` decode (hybrid 24 GatedDeltaNet + 8 full), `qwen3_5_vision` tower, M-RoPE + YaRN, **128 K validated context**, the experiment ledger, parity harness, recipe/register/publish integration.

**Device scope:** **Mac-only for v1** (9 B int8 ≈ 9.8 GB > the ~6.4 GB iPhone jetsam ceiling; Ornith-9B is Mac-only for the same reason). iPhone (int4-body + int8-head, ~6.5 GB) is a follow-on. GPU compute-unit only — the fp32 recurrence blocks ANE.

**Non-goals (v1):** MTP (weights ignored on load by the reference itself — `_keys_to_ignore_on_load_unexpected=[r"^mtp.*"]`); 256 K / 1 M context (KV-compression follow-on); `qwen3_5_moe`; iPhone ship; the ecosystem-wide provenance standard (this lane pilots it, §14).

## 2. Fidelity policy

Replicate the model's math exactly. Bounded knobs (device, context ceiling, quant) are ledgered optimizations, never silent compromises.

- **Gated linear attention (GatedDeltaNet):** exact.
- **Full-attention output gate:** exact — `attn_output_gate: true` means `q_proj` is **doubled** (`num_heads·head_dim·2 = 8192`), split into query + gate, and `attn_output *= sigmoid(gate)` before `o_proj` (`modeling:727,753-756,786`). Qwen3-Next-style; omit it and parity fails outright.
- **qk-norm:** per-head `RMSNorm(head_dim=256)` on Q and K **before** RoPE (`modeling:738-739,758-759`).
- **M-RoPE:** preserved (`mrope_interleaved`, `mrope_section [11,11,10]`) — the zoo's `qwen3_5.py` currently **collapses M-RoPE to 1-D**; re-enabling it on the hybrid backbone is genuinely-new (R3).
- **YaRN:** replicated (`rope_type yarn`, `factor 4.0`, `original_max 262144`, `theta 1e7`), including the `attention_scaling` **mscale** (≈1.139) that multiplies cos **and** sin.
- **Two distinct RMSNorm conventions** (must not conflate): text/qk/final norms use `(1 + weight)` (init 0); the DeltaNet `RMSNormGated` uses `weight` directly (init 1), normalizes per-head over `head_v_dim=128`, gates with `silu(z)`.
- **fp32** in conv, the recurrent/chunk scan, the decay gate `g`, and all RMSNorms (the reference's fallback runs these fp32 regardless of model dtype).
- **Vision:** native dynamic resolution is the *goal*; R1 may force resolution-bucketing — a ledgered fidelity delta, not a silent one.
- **Quant** is ledgered: **int4 is a candidate, not assumed-failing** — Ornith-9B (identical arch) passes int4lin 24/24 exact; the vocab head must be **absmax symmetric, per-block-32** (clipping crushes the 248 K-vocab head).

## 3. Architecture reality (grounded, verified)

Dense hybrid. `text_config`: hidden 4096, 32 layers, `head_dim` 256, GQA 16 Q / 4 KV, intermediate 12288, vocab 248320, `rms_norm_eps` 1e-6, SiLU, untied. `image_token_id` 248056, `vision_start_token_id` 248053 (span-counting bookkeeping, not splice), eos `[248046,248044]`, pad 248044.

`layer_types` = 8 × `[linear,linear,linear,full]` → **24 GatedDeltaNet + 8 full** (`full_attention_interval 4`).

**Memory:** only the 8 full layers grow a KV cache; the 24 linear layers carry fixed-size state → long-context cost ~4× lower than a dense transformer.

## 4. Context strategy

- **v1 headline (validated + published): 128 K**, within native 256 K (`original_max_position_embeddings 262144`); export parameterized.
- **Prefill is the real constraint (R2), not memory.** The `GatedDeltaUpdate` `scf.while` composite **does not lower on any device delegate**. Prefill must use a **loop-free** form: the in-graph loop-free chunk (fp16-unstable at chunk ≥ 64 — safe only at ~8) or the **fp32 Metal chunk kernel (GPU-only, chunk ≤ 64)**. Chunkwise-parallel GDN prefill at 128 K is the zoo's acknowledged open problem — treat as an explicit deliverable, not a given.
- **256 K** (native ceiling) and **1 M** (YaRN) are follow-ons gated on KV-cache compression for the 8 full layers.

## 5. Module decomposition

Vendor the zoo overlay into `models/qwen3_5/` and add the new units. Each unit: one purpose, typed interface, testable in isolation.

| # | Module | Purpose | Status |
|---|--------|---------|--------|
| 1 | `spec.py` | `config.json` → typed `Qwen35Spec` (layer_types, dims, rope/yarn/mrope, partial_rotary, vision geometry, token ids). | new (thin) |
| 2 | `qwen3_5_core.py` | Vendored zoo overlay: GatedDeltaNet + full-attn (incl. qk-norm, output gate, partial M-RoPE), 4-state layout, loop-free lowering. | **port** |
| 3 | `vision.py` | `qwen3_5_vision` tower: Conv3d patch embed, 2-D vision RoPE, **learned abs-pos embed (2304×1152) bilinearly interpolated**, cu_seqlens full attention (no windowing), patch merger → out_hidden 4096. Dynamic-res or bucketed (R1). | **new (hardest)** |
| 4 | `export.py` | Assemble embed + vision + decoder → `.aimodel` via `torch.export` → `coreai_torch.TorchConverter` + `get_decomp_table` (pi0/groot idiom). Bespoke; run **manually** (§9). | new |
| 5 | `parity.py` | Two-venv greedy A/B/C + Gate D (state cosine); margin-rule + rollout for long context (§11). | new (from `models/vlm/parity.py`, re-enabling M-RoPE) |
| 6 | recipe + `register` + `publish` | Recipe YAML (§13), `coreai-fabric register --catalog-path`, publish gate (apache-2.0 passes). | config |
| 7 | `ledger.py` + `provenance/*.jsonl` | Cross-cutting experiment ledger (§10). | new |

## 6. Core decode — the two attention types

Source verified against `modeling_qwen3_5.py`; the deployable lowering is the zoo's `qwen3_5.py` overlay.

**Linear — GatedDeltaNet (24 layers, `:512`):**
Dims: `linear_key_head_dim` 128 × 16 → key_dim 2048; `linear_value_head_dim` 128 × 32 → value_dim 4096; `conv_dim = key_dim·2 + value_dim = 8192`; `conv_kernel` 4; `num_v // num_k = 2` → q,k `repeat_interleave(2)`.
Decode step: `in_proj_{qkv,z,b,a}` → **causal_conv1d_update** (depthwise, on the concatenated 8192, kernel 4, SiLU) → split q/k/v → gates `β=σ(b)`, `g=−exp(A_log)·softplus(a+dt_bias)` (fp32) → **delta rule with decay applied to the state first**: `S' = decay(g)·S_{t-1}` (decay = per-head **scalar**); `S_t = S' + k⊗[β(v − k·S')]`; `out = q·S_t`, with **in-kernel L2-norm(q,k)** (eps 1e-6) and **query scale `1/√128`** → `RMSNormGated(out, z)` → `out_proj` (4096→4096).

**Full (8 layers, `:714`):** q_norm/k_norm(256) on Q,K → **partial M-RoPE** (`rotary_dim = 0.25·256 = 64`; `q_pass`/`k_pass` 192 dims untouched; YaRN + mscale) → softmax GQA (16/4, `scaling 256^-0.5`) + KV cache → **sigmoid output gate** (doubled q_proj) → `o_proj`.

## 7. State layout (deployable — B1)

**Do not emit per-layer named states** (56 states won't load). Pack into **4 layer-major tensors** (the shipped `qwen3_5.py` `build_decode_state` / `DECODE_STATE_NAMES` layout), = 2 KV + 2 extra, within the pipelined engine's ≤2-extra limit:

- `keyCache`, `valueCache` — the 8 full layers.
- `convState` `[n_linear=24, 1, conv_dim=8192, kernel-1=3]` (engine rolling-window packing; the transformers ref stores width=kernel=4).
- `recState` `[n_linear=24, 1, num_v=32, dk=128, dv=128]`.

Carried as in-place **states** (RWKV7/qwen3_5 idiom, faster) on the patched pipelined engine; pi0fast's tensor-I/O `TensorCache` is the fallback idiom. Per-layer slice via the overlay's `SSMState.update_states(lin_idx, …)`.

## 8. Vision tower (the primary new risk — R1)

`Qwen3_5VisionModel` (`:1088`): depth 27, hidden 1152, 16 heads, patch 16, merge 2, temporal 2, out_hidden **4096** (Qwythos override; config default 3584 — confirm). Mechanisms the exporter must replicate (all verified in the reference):
- **Conv3d** patch embed over (temporal 2, 16, 16); block-major 2×2 token ordering the merger depends on.
- **2-D vision RoPE** (`Qwen3_5VisionRotaryEmbedding`, applied fp32).
- **Learned abs-pos embedding** `Embedding(2304, 1152)` (48×48 grid) **bilinearly interpolated** onto the dynamic grid and **added** to patches (`fast_pos_embed_interpolate`).
- **cu_seqlens full attention** — all 27 blocks, **no window attention** (unlike Qwen2.5-VL); no full-attention-index list.
- **Patch merger** `LN(1152) → fc1(4608→4608) → GELU → fc2(4608→4096)`.
- **No deepstack** (`deepstack_visual_indexes` empty; genuinely absent).

**R1:** every shipped tower is fixed-grid/fixed-slice; a dynamic-shaped-output graph the runtime cannot execute. **Spike S3** decides: true dynamic-res vs **resolution-bucketing** (a set of fixed grids, the only proven export shape) — measure and ledger the fidelity delta rather than assume a single dynamic-res graph exports.

## 9. Bundle, host orchestration, and the manual convert step

Bundle: **vision** (`.aimodel`) + text decoder (`.aimodel`) with embed. Host: vision-encode → id-space embed gather → **splice image features at `image_token_id` (248056)** via `masked_scatter` (image_embeds as a static buffer) → prefill → greedy decode with the 4 states. Precedent: `qwen3_vl.py` (VLM 3-graph rider) + MiniCPM-V-4.6 (id-space splice + static image buffer + 4 hybrid states, proven on the pipelined engine).

**Fabric does not run `.py` tools.** `convert.py:194 is_script_tool` makes `convert`/`run` **refuse and print a manual hint**. So: the operator runs `models/qwen3_5/export.py` by hand, drops the bundle at `build/<id>/<id>.aimodel`, then `coreai-fabric verify <id>` (and `coreai-fabric run <id> -- <cmd>` to log the manual run to `attempts/<id>.jsonl`). The lane's credibility rests on its own `export.py` + `parity.py` (pi0/groot idiom), not on fabric-driven convert.

**Runtime dependency:** Apple's `coreai-models` + the zoo's `apps/coreai-pipelined-extra-states.patch` (+ the static-inputs patch for the VLM), surfaced via CoreAIKit / `ChatAdapter` / `VLMAdapter`. GPU compute-unit only.

## 10. Provenance / experiment ledger (pilot discipline)

Every optimization attempt (quant, KV compression, YaRN, vision-res bucketing, loop-free-prefill variants) → an immutable, committed, machine-readable record in `models/qwen3_5/provenance/*.jsonl`, linked from the model card and the catalog artifact provenance. **Negative results recorded too.**

```json
{"id":"exp-0007","hypothesis":"int4lin body holds 24/24 like Ornith-9B",
 "target":{"model":"qwythos-9b","weights_rev":"763f72f…","component":"gdn+full body"},
 "config_hash":"sha256:…","seed":0,
 "env_fingerprint":{"os":"…","py":"…","transformers":"<pinned>","coreai_torch":"0.4.1","coremltools":"9.0"},
 "deltas":{"gate":"24/24","first_tok_cos":0.9997,"rollout_margin":0.31,"tok_s":59,"bundle_gb":7.5,"peak_mem_gb":9.8},
 "verdict":"candidate","why":"matches Ornith int4lin; needs 128K rollout sanity",
 "repro_cmd":".venv/bin/python models/qwen3_5/parity.py --compare --exp exp-0007 …"}
```

Consumed/gated by fabric's publish provenance + privacy guard (no local paths, no hardware fingerprints — existing `publish.py:203-215`).

## 11. Parity & verification

Two-venv (reference `.venv-lerobot` transformers → npz; `--compare` drives the asset). **Pin/verify the reference transformers** (config declares 5.12.1; venv is 5.3.0) and confirm Qwythos `rope_parameters` parse identically before trusting numbers. Note the base `models/vlm/parity.py` **strips M-RoPE** (`:58-64`) — re-enabling it is required and changes what Gate B certifies.

- **A** greedy token-exact over decode_len (PRIMARY) — target the engine 24/24 / 12/12 bar Ornith hit.
- **B** first-token logit cosine + argmax — realistic ≥ 0.999 for prefill/short decode.
- **C** vision-feature cosine (coreai vs torch) — self-validates preprocessing.
- **D (new)** conv_state + recState cosine vs the reference `_gated_delta_step` (bit-identical at S=1) — isolates the DeltaNet lowering.
- **Long context (128 K):** bit-exact ≥ 0.999 is **not** the gate (fp16 noise compounds). Use the zoo's **margin rule + rollout sanity** (logit margin at each step + a full-rollout coherence check), as Ornith does.

## 12. De-risk spike order (empirical; each writes ledger entries; stop-on-red)

1. **S1 — vendored decode gate:** run the ported overlay on Qwythos; reproduce Ornith-style fp16/int8hu/int4lin gate + engine token-exact; confirm the **GVA 32v/16k** branch and the 4-state layout load. (Highest-confidence; mostly a port.)
2. **S2 — M-RoPE on the hybrid backbone:** re-enable interleaved 3-D M-RoPE + YaRN mscale on the decoder; first-token logit cosine vs torch (R3).
3. **S3 — vision (primary unknown):** dynamic-res vs resolution-bucketing for `qwen3_5_vision`; measure export-ability + fidelity delta (R1).
4. **S4 — 128 K prefill:** loop-free chunk vs fp32 Metal kernel; TTFT + rollout margin (R2).
5. **S5 — quant ship shape:** int8hu vs int4lin under the ledger (int4 is a live candidate).

## 13. Fabric ↔ catalog integration (validated)

**Recipe `recipes/qwen3_5-qwythos-9b.yaml`** — required top-level `[id, upstream, conversion, expected, parity, publish, status]` (schema `recipe.schema.json`). `conversion` needs `[tool, quantization, precision]`; `upstream` `[hf_repo, license, license_terms]`; `parity` `[gate_a, gate_b]`; `publish` `[hf_target_namespace: kevinqz, repo_name]`. Include a `catalog:` block (needs `[name, family, capabilities, modalities, bundle_kind, runtime_facts]`).

**`coreai-fabric register --catalog-path <clone>`** appends model+artifact+source entries, bumps counts, and **replays the catalog's full CI locally** before opening the PR. The catalog entry must satisfy (validated against the live schema/audit):
- `source_group: fabric` (model) **+** artifact `group: external` **+** `officiality.apple_export_recipe: false` — the fabric↔catalog contract (audit `ALLOWED_GROUP_PAIRINGS`).
- `architecture: transformer` (the enum rejects "hybrid"/"linear-attention"); put "hybrid 24 GatedDeltaNet + 8 full-attention, qwen3_5_vision" in `notes`.
- `capabilities: [vision-language, hybrid-llm, reasoning, agentic]` — `vision-language` derives `bundle_kind: vlm`; `hybrid-llm` marks the arch (recognized by `exports.py`). **Drop `function-calling`** (alias of `agentic`; a literal value spawns an orphan task page). **Do NOT use `catalog.traits`** — `register` would emit a top-level `traits` key that fails the catalog's `additionalProperties:false` (B2).
- **No `framework_contract`** (lerobot-only enum; stays null for a VLM).
- New id `qwythos-9b` (edit-distance-safe from `qwen3-5-*`; a qwen3.5 *text* LLM already exists but Qwythos does not — not an `alternate_artifacts`).
- **License-laundering guard (audit cat.10):** for `commercial_use: likely` on apache-2.0, add/verify an `empero-ai/Qwythos` `original_model_sources` upstream with `license_terms: permissive` and verified `trust`/`owner`; else it fails.
- `context_window: 128K`, `streaming: true` — accepted, no cap. No content/appropriateness gate exists (the catalog already lists an abliterated model); the "Claude-Mythos" name is a trademark/governance judgment, not a schema block — flag to the curator.

**Publish gate (all confirmed):** apache-2.0 ∈ permissive allowlist → passes; restricted = hard-refuse; bundle allowlist refuses raw `*.safetensors`/`*.pt` (weights live in `main.mlirb`); privacy guard refuses local paths + hardware fingerprints. **R6:** the ~10–18 GB bundle upload uses `api.upload_folder` with no resumable/sharded path — on a throttled/Xet network this is a practical risk; consider a robust sharded upload before shipping.

## 14. Follow-on (separate specs)
1. Ecosystem-wide provenance standard (generalize §10 across catalog + fabric + zoo).
2. 256 K / 1 M context via KV-cache compression (8 full layers).
3. iPhone ship (int4-body + int8-head, ~6.5 GB).
4. MTP speculative decode; `qwen3_5_moe` lane.

## 15. Resolved / open questions
- **Runtime carries arbitrary named recurrent state? — YES** (4-state layout ships in Ornith/MiniCPM-V-4.6; `HybridCoreAIEngine` binds any `stateNames`).
- **DeltaNet lowers without a custom kernel? — YES for decode** (standard ops), **NO for the `while_loop` composite** → lower the loop-free forms only.
- **Graph-building mechanism? — `torch.export` → `coreai_torch.TorchConverter`** (coremltools underneath), not "plain Core AI ops"; budget for `coreai_torch` op-lowering gaps (precedent: missing `aten.bucketize`/`empty_permuted` workarounds).
- **Catalog `bundle_kind`? — `vlm` suffices**; no new kind.
- **Open:** does dynamic-res vision export at all, or must we bucket (S3)? Can 128 K loop-free prefill hit an acceptable TTFT (S4)? Does Qwythos's specific RL fine-tune preserve Ornith's int4 tolerance (S5)?
