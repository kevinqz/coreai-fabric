# Class-B v0.6.0 build specs — FastWAM · EO-1 · MolmoAct2

The three remaining whole-model VLAs. All techniques are proven (see
`v060-conversion-findings.md`); this file is the executable spec so each build is
mechanical, not investigative. Shared recipe:

> **int8 combined driver** (proven on `lingbot_va`, scratchpad `lva_int8.py`):
> build model → load weights → `quantize_(model, Int8WeightOnlyConfig())` (torchao
> 0.17) → **delete the source safetensors** (head stays in RAM) → `torch.export` →
> `to_coreai` → `save_asset` (~1 byte/param) → parity from the in-RAM head.
> Always: stub `lerobot.utils.import_utils` (`_diffusers_available=True`), rewrite
> any `torch.polar`/`view_as_complex` RoPE to real `cos`/`sin`, keep
> `attn_mode="torch"` (SDPA, never flex). Gate B = `graph_output_cosine`.

---

## FastWAM — `lerobot/fastwam_base` (apache-2.0, 6B, downloaded to `build/_fastwam/`)

**Deployable graph:** the action-denoise-step, `FastWAM._predict_action_noise_with_cache`
= `action_expert.pre_dit` → `mot.forward_action_with_video_cache` → `action_expert.post_dit`.
The **video expert is NOT in the asset** — its per-layer K/V arrive as a
`video_kv_cache` graph input (host prefills once via `video_expert.pre_dit` +
`mot.prefill_video_cache`).

- **`ActionDiT`** (`wan/modular.py:70`) — action_dim 7, hidden_dim 1024, ffn_dim 4096,
  num_heads 24, attn_head_dim 128 (inner 3072), num_layers 30, text_dim 4096,
  freq_dim 256. `forward` is self-contained (self-attn + text cross-attn) but the
  TRUE inference path is the MoT-with-video-cache one.
- **`MoT`** (`wan/modular.py:600`) shares the experts' `.blocks` (no own params);
  requires BOTH `video` + `action` mixtures to construct. `forward_action_with_video_cache`
  (`:737`) runs action tokens through each `MoTLayer` in `mode="action_cached"`,
  cross-attending to `video_kv_cache[layer]["k"/"v"]` (30 layers).
- **Graph inputs:** `latents_action [B, action_horizon, 7]`, `timestep [B]`,
  text `context [B, L, 4096]` + `context_mask`, `video_kv_cache` (30 × {k, v}),
  `attention_mask`, `video_seq_len`. Weights: `model.action_expert.*` (+ build a
  minimal/random `video_expert` only to satisfy `MoT.__init__`, since its K/V come
  from the cache input).
- **RoPE:** `precompute_freqs_cis`/`apply_dense_rope` (`wan/video_dit.py:198,202`)
  use `view_as_complex` → rewrite real (the lingbot_va fix).
- **Size:** action path ~1B → fp16 ~2GB likely loads without int8; int8 ~1GB as
  fallback. `action_scheduler`: 20 steps, flow-matching, shift 5.
- **Parity:** seeded synthetic `video_kv_cache` + text context, both sides; the
  metric gates the action-DiT export (host owns the video prefill).
- **⚠️ BLOCKER (version skew) — resolve before building.** The lane is ~90% built
  (`scratchpad/fastwam_int8.py`: lerobot stub + real-RoPE + MoT construction with a
  tiny video stub + int8 + the action-denoise-step wrapper; torch forward runs,
  output `[1, 32, 7]`). BUT the `lerobot/fastwam_base` checkpoint's action expert
  (`model.mot.mixtures.action.*`, 820 tensors) contains ONLY `blocks`,
  `text_embedding`, `time_embedding`, `time_projection` — it has **no
  `action_encoder` (7→hidden), no `head` (hidden→7), and no `proprio_encoder`**
  (grep: zero action_encoder keys, zero dim-7 tensors, all keys under
  `model.mot.mixtures`). The current `main` `ActionDiT.__init__` DOES create
  `action_encoder`/`head`, and `_build_core_model` references a `proprio encoder`,
  so loading this checkpoint into today's code leaves the action I/O projections
  RANDOM → the shipped model would be garbage (the parity would still read ~1.0
  because both sides share the same random weights — a trap).
- **RESOLVED (evidence-based):** not a code skew (main == v0.6.0 both create
  `action_encoder`/`head`). `lerobot/fastwam_base` is the BASE world-model with **no
  trained action heads** (zero dim-7 tensors). The deployable action policy is a
  FINETUNED variant: **`lerobot/fastwam_libero_uncond_2cam224`** (1652 keys) HAS
  `model.mot.mixtures.action.action_encoder.*` + `head.*` — verified via a
  safetensors range-request header read (no full download). Convert the LIBERO
  variant (apache), not the base; the 90%-built lane (`scratchpad/fastwam_int8.py`)
  needs only the weights swap. Other variant: `fastwam_robotwin_uncond_3cam_384`.

## LingBot-VLA 2.0 — `robbyant/lingbot-vla-v2-6b` (Apache-2.0, 6.38B) — FEASIBILITY PROVEN, build in progress

**The MoE export blocker is SOLVED (in principle) — the dense path already exists in
the upstream code.** `Qwen2TokenMoeBlock.forward` has an eager `else` branch
(`_moe_implementation != 'fused'`) that runs ALL experts and combines with a one-hot
routing mask — pure torch (`stack`/`one_hot`/`einsum`/`topk`/`softmax`), no group_gemm,
no data-dependent gather. The checkpoint stores the experts as fused 3D params
(`experts.gate_proj [E, inter, hidden]`), so the deployable graph rewrites that eager
combine as a dense einsum over the 3D weights — mathematically exact, export-friendly.

**Architecture (inferred from checkpoint shapes — the HF `config.json` is a bare
`{"vlm_family":"qwen3_vl"}` wrapper; real config comes from the framework yaml +
weight shapes):** a pi0/EO-1-style **coupled shared-attention VLA**. Qwen3-VL-4B
backbone + a parallel `qwen_expert` transformer; per layer, BOTH stacks compute q/k/v,
they're concatenated, joint attention runs, then each stack applies its own o_proj +
MLP. The action denoise runs in the expert stack conditioned on the VLM's cached
prefix K/V — the EO-1/MolmoAct2 split (KV as graph inputs), plus MoE MLPs.

- **qwen_expert:** 36 layers, ALL MoE. hidden 768; attention 32 q-heads / 8 kv-heads
  (GQA) × head_dim 128 (q_proj 768→4096, k/v 768→1024); MoE 32 experts, top-4,
  intermediate 512, **sigmoid** router + `routed_scaling_factor` 4.0 +
  `e_score_correction_bias` (loss-free load-balance bias added to scores before top-k;
  weights gathered from the *pre-bias* sigmoid scores); shared expert (intermediate
  704) with a sigmoid gate; `adanorm_time` (time embedding drives AdaNorm on the
  input/post-attn norms). ~7.15GB fp32 (901 tensors).
- **Deployable graph** = `FlowMatchingV2.predict_velocity`: `embed_suffix(state, x_t,
  timestep)` → suffix_embs + time (ada_cond); joint forward with
  `inputs_embeds=[None, suffix]` (VLM branch None — its K/V come from
  `past_key_values`), so only the expert computes q/k/v, the cached prefix K/V is
  prepended, attention runs, o_proj + AdaNorm + dense-MoE per layer; final norm →
  `action_out_proj` on the last `n_action_steps` → velocity. Host owns the Qwen3-VL
  prefix (`embed_prefix`), the Euler loop (`sample_actions`), un-normalization.
- **Attention = eager.** The deploy policy (`deploy/lingbot_vla_v2_policy.py`) itself
  sets `attention_implementation='eager'` for inference → `our_eager_attention_forward`
  (manual `softmax(QKᵀ/√d + mask)·V`). No SDPA at all → the mask-free
  FoldMultiplyIntoSDPAScale segfault (macOS 27) is structurally avoided (better than
  the bool-keep-mask workaround).
- **License: Apache-2.0** (explicit README prose: "This project is licensed under the
  Apache-2.0 License") → PUBLISHABLE (unlike MolmoAct2). Set the recipe license to
  apache-2.0 citing the README.
- **Weight fetch:** need only the qwen_expert + action heads (~7.3GB), NOT the 17.75GB
  Qwen3-VL backbone (host-side). The qwen_expert tensors are near-contiguous per shard
  (1–2 spans each across the 6 shards) → fetch one envelope range per shard, 6 parallel
  downloads, reconstruct `qwen_expert.safetensors`.
- **Size:** ~1.8B expert → int8 ~1.8GB (fp32 7.15GB / fp16 3.6GB); int8 to be safe on
  ANE. Standalone-import the qwen2_action_expert classes + joint-forward (VLM=None
  path) + embed_suffix, monkeypatch the MoE combine to the dense 3D einsum.
- **RESULT (2026-07-08): SOLVED — LOADS on ANE + Gate B PASSES via graph-split.**
  min graph_output_cosine **0.99948** / median 0.99978 (8 obs), recipe `verified`,
  publish dry-run clean (Apache-2.0). The monolithic 36-layer dense-MoE hit the ANE
  single-program limit (0x10004, ~1.5GB graph-complexity ceiling: fp16 L=12/14 load,
  L=18=1.79GB fails; int8 L=24=1.48GB loads). Fix: split the action expert into **4
  loadable ANE programs** — a tiny `embed` + 3×12-layer fp16 blocks (~1.19GB each). The
  host chains `embed→block0→block1→block2` and loads the big blocks **sequentially**
  (load→run→free — they must not be co-resident in ANE memory). Bundle =
  `lingbot-vla-v2.aimodel/{metadata,manifest}.json + programs/{embed,block0,block1,block2}.aimodel`;
  `manifest.json` declares the chain + per-block prefix-K/V slices. fp16 chosen over int8
  (int8 experts dropped worst-obs to 0.9982 < gate; fp16 blocks fit the ceiling). Driver:
  `models/lingbotvla/export_split.py` (N-block). Two reusable ANE lessons: (a) 0x10004 is
  a per-program graph-complexity ceiling, not disk/byte-size; (b) separate assets loaded
  sequentially beat a multi-entrypoint asset (whose `AIModel.load` makes all programs
  co-resident and re-crosses the limit).

  ### (superseded) op-coverage-only result
  The whole
  lane was built and run: standalone module (norm/attn/decoder verbatim, MoE dense
  einsum over the fused 3D weights), 911 weights loaded strict-clean, fp32 torch ref
  runs (velocity [1,32,55]). Export SUCCEEDS on coreai_torch 0.4.1 (fp16 3.58GB /
  int8-experts 2.23GB). MoE dense-fusion is mathematically EXACT vs the sparse top-4
  reference (cosine 1.0, maxdiff ~1e-4). A **2-layer int8-experts model LOADS on the
  ANE and runs** (graph_output_cosine 0.999926) — every op lowers + executes (topk,
  one_hot, gather, 3D batched einsum, eager softmax attention, repeat_interleave GQA,
  AdaRMSNorm FiLM, manual int8 dequant). **BUT the full 36-layer asset fails ANE load
  with Program load failure `0x10004`** at BOTH fp16 (3.58GB) and int8-experts (2.23GB)
  — under FastWAM's 4GB int8 that loads fine, so this is a **graph-COMPLEXITY limit**
  (36 layers × 32-way dense-MoE einsums = a very large op/constant count), not a byte
  limit. Purging `coreai-cache` (11GB→38GB free) did not help — it's structural.
  **The conversion is proven feasible; only the single-program full-depth load is
  blocked.** Fixes: a sparse-gather export (needs coreai_torch static top-k gather), a
  graph-split (ship N sub-graphs of fewer layers, host chains the residual stream), or
  an ANE program-size raise. Apache-2.0 → publishable once loadable. Findings:
  `build/lingbot-vla-v2/op-coverage-findings.json`.

  This mirrors the GR00T milestone (op-coverage proven, not a hosted artifact) but for a
  different reason: GR00T is license-gated, lingbot-vla-v2 is ANE-program-size-gated. It
  is the single largest conversion in the set — a 36-layer MoE joint-attention stack.

### (original notes retained below)
## LingBot-VLA 2.0 — `robbyant/lingbot-vla-v2-6b` (Apache-2.0 per README, 6.38B) — PRIORITY

Brand-new (2026-07-07) robbyant VLA foundation model — NOT the LeRobot v0.6.0
`lingbot_va` (that's a separate Wan video-model, already done) nor the `lingbot-vision`
ViTs. **Coupled / whole-model, same pattern as EO-1**: weight map is all `model.*`
(1708 tensors) with only tiny `model.action_in_proj`/`action_out_proj`/
`action_time_mlp_*` — the action denoise runs INSIDE the Qwen3-VL-4B backbone via
action tokens, no separable head. Backbone Qwen3-VL-4B-Instruct + depth (moge) +
dino_video experts. License field on HF is empty but the repo README carries an
explicit Apache-2.0 badge + `LICENSE` (like MolmoAct2-LIBERO) → publishable.
**Build:** whole-VLM int8 (~3.4GB from 13GB fp16). **⚠️ HARDER than EO-1 — it's a
MoE VLA.** The action expert (`lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py`)
is a **Qwen2-MoE** (`Qwen2FusedExperts`, group_gemm, dynamic top-k routing +
`_update_moe_runtime_stats`), plus `flex_attention.py`, a Qwen3-VL-in-VLA patch, and
MoGe (depth) + dino_video vision experts — all in robbyant's own `lingbotvla`
framework (not LeRobot/transformers). **MoE dynamic routing is the export blocker**
(data-dependent expert gather is coremltools-hostile): it must be made static
(fixed routing / dense-fuse the experts) before export. flex_attention → SDPA (the
config may allow a torch/SDPA mode like Wan did — check first). This is a
research-grade conversion, NOT a mechanical int8 lane. Recommend: attempt AFTER the
clean LeRobot coupled-VLMs (EO-1, MolmoAct2), or descope to a fixed-routing export.
Priority was user-set (2026-07-07) but the MoE complexity reorders it realistically.

## EO-1 — `IPEC-COMMUNITY/EO-1-3B` (MIT, 3.77B)

**Whole-model / coupled** — no separable head. `EO1VisionFlowMatchingModel`: the
action denoise runs INSIDE the Qwen2.5-VL backbone via `action_token_id` 151666 /
`action_pass_id` 151672 (`num_action_layers` 2 reuse VLM layers). Weight map =
`vlm_backbone` (824 tensors) + tiny `action_in_proj`/`action_out_proj`/
`action_time_mlp`/`state_proj`.

- **Deployable graph:** the full Qwen2.5-VL forward on the action-token positions →
  the action projections. `action_chunk_size` 16, `max_action_dim` 32,
  `num_denoise_steps` 10. This is a ~7.5GB fp16 model → **int8 mandatory** (~3.8GB).
- MIT → publishable. Standard `transformers` Qwen2.5-VL (may need `trust_remote_code`
  for the `eo1` model_type / custom modeling). Build via the HF class, int8, export
  the action-token denoise pass with a fixed action-token layout.

## MolmoAct2 — `lerobot/MolmoAct2-LIBERO-LeRobot` (5.44B) — DONE (index-only)

**CORRECTION to the original assumption.** I expected a coupled whole-VLM (like
EO-1). It is NOT — MolmoAct2's action expert is **SEPARABLE**, and the deployable
graph is JUST the small action expert (the **FastWAM pattern**, conditioned on a VLM
instead of a video expert). This was the single most important finding: it turned a
projected ~2.7GB whole-VLM int8 build into a ~1.16GB fp16 action-only build.

- **Arch:** Molmo2-ER VLM backbone + a flow-matching continuous action expert joined
  by **per-layer KV conditioning** — but the expert is a proper `nn.Module`
  (`ActionExpert`, 36 blocks, hidden 768, 8 heads, ~578M params), one block per VLM
  layer, each cross-attending to that layer's VLM K/V. `num_flow_timesteps` 8,
  `chunk_size`/`n_action_steps` 10, `max_action_dim` 32, VLM `llm_kv_dim` = 8·128 =
  1024, VLM 36 layers.
- **Self-contained modeling.** The LeRobot policy package bundles the full HF modeling
  (`lerobot/policies/molmoact2/molmoact2_hf_model/`, Apache-2.0 code header) — no
  allenai `trust_remote_code` download. The action-expert classes are pure torch
  (no transformers dep beyond the VLM parts), so we **vendor just those classes** into
  `models/molmoact2/action_expert.py` and never instantiate the 5.44B VLM.
- **Deployable graph** = `ActionExpert.forward(noisy_actions[B,10,32], timestep[B],
  ctx_k, ctx_v)` → velocity, where `ctx_k/ctx_v` are the VLM's per-layer K/V stacked
  `[36, B, ctx_seq, 1024]` (the FastWAM vk/vv trick). The expert projects each layer's
  K/V (`context_k/v_proj` 1024→768) and cross-attends internally. Host owns the VLM
  prefill (`collect_layer_kv_states=True`), the Euler loop (`trajectory += dt·velocity`,
  8 steps), un-normalization (`norm_stats.json`, `norm_tag="libero"`).
- **Parity is honest.** Verified by reading the upstream loop: `ActionExpert.forward`
  (modulation=None) is *mathematically identical* to the inference loop's per-step
  `forward_with_context` — the block/final layers compute
  `self.modulation(conditioning).chunk(...)` internally when modulation is None, which
  is exactly what `prepare_modulation_cache` precomputes. No random-weight trap.
- **fp16, no int8.** ~578M params → fp16 ~1.16GB loads on the ANE directly. RoPE is
  already real cos/sin (no complex rewrite). SDPA runs mask-free (causal_attn=False) →
  inject a **data-dependent all-true bool keep-mask** (`k.abs().sum() >= -1`, expanded)
  so the mask-free `FoldMultiplyIntoSDPAScale` segfault can't trigger. (Bool keep-mask,
  like FastWAM's — NOT an additive float mask, which folds.)
- **Targeted weight fetch.** Don't download the 11GB checkpoint — the 588
  `action_expert.*` tensors are **contiguous** in the single `model.safetensors`
  (2.31GB span). Read the header offsets, range-fetch the one contiguous span (8
  parallel chunks to beat CDN throttling), reconstruct a clean `action_expert.safetensors`.
- **⚠️ LICENSE → INDEX-ONLY.** The upstream license is **unpopulated everywhere** (the
  lerobot mirror, `allenai/MolmoAct2-LIBERO`, and base `allenai/MolmoAct2` all have no
  `license:` field, no LICENSE file, no README badge as of 2026-07-08 — only the older
  v1 `allenai/MolmoAct-7B` is apache-2.0). My earlier "LIBERO is explicitly apache"
  claim did NOT hold up. An unpopulated license is not an affirmative redistribution
  grant, so `coreai-fabric validate` flags it restricted and publish refuses the
  weights path. **Deliverable = the reproducible recipe + measured Gate-B parity**
  (like GR00T). Revisit for publish once AI2 populates the license.
- One conversion path serves the 4 non-Think variants (identical arch); Think adds the
  depth-reasoning branch.
- **RESULT (2026-07-08):** fp16 asset **1.16GB**, Gate A + Gate B PASSED,
  `graph_output_cosine` min **0.99999681** / median **0.99999941** over 8 obs
  (threshold 0.999). Recipe `molmoact2-libero` status → verified. Publish correctly
  refused (restricted license); terminal state = the verified reproducible recipe in
  fabric (index-only, no catalog artifact — the GR00T precedent).

---

## ⚠️ Disk: purge the ANE compile cache

The CoreAI runtime caches every compiled `.aimodel` under
`~/Library/Caches/coreai-cache` — it ballooned to **53 GB** mid-session and looked
like a hard disk wall (only 12 GB `df`-available on a 926 GB disk). `rm -rf
~/Library/Caches/coreai-cache` reclaimed it → 65 GB free, and the int8 exports fit
again. **When disk gets tight during a run, purge `coreai-cache` first** — it's the
usual culprit, not the checkpoints. Also `~/.cache/uv` (torchao/pip) and macOS local
snapshots (`sudo tmutil deletelocalsnapshots /`) if still needed.

## Order & disk

Do FastWAM first (smallest deployable graph — action path only, likely fp16). Then
EO-1 (MIT, int8). Then MolmoAct2 (apache LIBERO, int8). Disk is the binding
constraint (~16-22GB free): one 7-12GB checkpoint at a time; the delete-after-load
trick frees the source before `save_asset`. Register each into the catalog off
current `main` (single clean PR each; PR #29 + #30 already establish the pattern).
