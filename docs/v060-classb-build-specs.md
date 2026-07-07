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

## MolmoAct2 — `lerobot/MolmoAct2-LIBERO-LeRobot` (5.44B)

**Whole-model / coupled** — Molmo2-ER VLM + a flow-matching action expert connected
by **per-layer KV conditioning** (tightly coupled, not a peel-off head). custom_code
(`auto_map`), `MolmoAct2ForConditionalGeneration`, `num_flow_timesteps` 8,
`chunk_size`/`n_action_steps` 10, `expected_max_action_dim` 32.

- **License:** the `lerobot/` mirror's field is empty; `allenai/MolmoAct2-LIBERO-LeRobot`
  (same weights) is explicitly **apache-2.0** → publish the LIBERO variant; the other
  four (DROID/SO100_101/Think/BimanualYAM) are index-only until AI2 populates the
  license, or adopt the LIBERO apache precedent.
- **Deployable graph:** the full VLM + action-expert flow-matching pass → int8
  (~2.7GB). One conversion path serves the 4 non-Think variants (identical arch,
  5,442,196,272 params); Think adds the depth-reasoning branch.

---

## Order & disk

Do FastWAM first (smallest deployable graph — action path only, likely fp16). Then
EO-1 (MIT, int8). Then MolmoAct2 (apache LIBERO, int8). Disk is the binding
constraint (~16-22GB free): one 7-12GB checkpoint at a time; the delete-after-load
trick frees the source before `save_asset`. Register each into the catalog off
current `main` (single clean PR each; PR #29 + #30 already establish the pattern).
