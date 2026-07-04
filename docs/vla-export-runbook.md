# VLA / action lane — pi0 export runbook

The ordered, gated hands-on session to convert a LeRobot VLA (pi0) to an Apple
`.aimodel`. Grounded against the real lerobot source + the fabric lowering chain
(see the toolkit spec in the workflow output). **Each STOP gate is a real stop —
do not spend disk/compute on phase N+1 until phase N's gate is green.**

## The two-venv model (why)
- **venv-A** = `.venv-lerobot` (torch **2.9.0** + `lerobot[pi]==0.5.1` → transformers 5.3.0,
  numpy <2.3). Runs `torch.export` → `.pt2`. Created by `scripts/setup_vla_export.sh`.
- **venv-B** = the fabric `.venv` (torch **2.9.0** + coreai_torch 0.4.1 + coremltools 9.0).
  Loads the `.pt2` and lowers it to `.aimodel`. Already exists.
- **Both on torch 2.9.0** → the serialized ExportedProgram never crosses a torch
  version (zero `.pt2` risk). The split is forced only by **transformers 5.3 vs 4.57**
  (a hard conflict), not torch.

## Normalization — the host owns everything (VERIFIED)
The graphs operate in **normalized, padded** space only. Everything else is host code:
resize-with-padding to 224², SigLIP does its own [0,1]→[−1,1] (VISUAL = IDENTITY);
STATE + ACTION use **MEAN_STD** (from `policy_preprocessor.json` / `policy_postprocessor.json`
→ shipped as `norm_stats.json`); tokenize to 48; un-normalize + un-pad the action chunk after.
`action_parity` compares in **normalized** space (never un-normalized — the pi0 analog of
greedy_parity comparing logits pre-detokenization).

---

## Phase 0 — pure reading (do NOW, zero install, zero disk)
1. Read lerobot `modeling_pi0.py` `make_att_2d_masks` / `_prepare_attention_masks_4d`
   (~:840-843). Confirm the suffix attention mask is a **dense tensor**, not in-place booleans.
   - **GATE 0:** if it's an in-place boolean assembly, add a mask-rewrite shim to
     `DenoiseWrapper` (models/pi0/export.py) before proceeding. (Highest-value read: it
     decides whether the LLM+action graph exports clean.)

## Phase 1 — prove the pipeline on a NON-VLM policy (de-risks everything not-pi0)
> An ACT/diffusion policy is ~0.2–1GB (fits this disk) and has NO vision tower — it
> exercises the two-venv dance + `.aimodel` packaging + the new `action_parity` runner
> without pi0's one real risk. **Do this before pi0.**
2. Land `run_action_parity` + `_bootstrap_ci` (parity_runner.py) and `setup_vla_export.sh`.
3. Convert an ACT policy (e.g. `lerobot/act_aloha_sim_transfer_cube_human`, apache, 0.21GB)
   end-to-end: export (venv-A) → lower (venv-B) → `.aimodel` → `action_parity` green.
   - **GATE 1:** if the runner or packaging is wrong, fix it here — cheaply — before pi0.
   - **DONE (2026-07-04):** ACT single-graph proven twice — `kevinqz/ACT-Aloha-{TransferCube,Insertion}-CoreAI`,
     `action_parity` ≈ 1.0. `models/act/{export,parity}.py` are the reference implementation;
     `verify` records the two-venv measurement through the standard path.

## Phase 1.5 — Diffusion Policy sampler lane (PROVEN — the direct pi0 rehearsal)
> A Diffusion Policy is the tiny, VLM-free analog of pi0: a **split export** (`encode` runs
> once + `denoise_step` the host drives N times) in ONE `.aimodel`, plus a host-side N-step
> sampler loop. Same shape as pi0's flow-matching, minus the 4B VLM + SigLIP. Proving it
> de-risks EVERYTHING about pi0 except the vision tower (Phase 2b).
1.5a `.venv-lerobot/bin/python models/diffusion/export.py export --repo lerobot/diffusion_pusht --out build/diffusion-pusht`
     then `--lower` in venv-B. Two entrypoints land in one asset via
     `TorchConverter.add_exported_program(..., entrypoint_name="encode"/"denoise_step")`.
     - **GATE 1.5a (op-coverage):** the denoise 1D-U-Net (Conv1d + GroupNorm + FiLM + sinusoidal
       step-embed) must lower on coremltools 9.0. Probe `--only denoise` FIRST (cheapest).
1.5b `action_parity` via `models/diffusion/parity.py`: the reference (torch encode+denoise) and
     the asset run the **identical deterministic DDPM host loop** (`_ddpm_update`, posterior mean,
     per-step variance = 0) — so the only difference is the exported graphs. Compares the NORMALIZED
     action trajectory (min chunk-cosine + per-dim MAE + bootstrap CI). This is the reusable host
     sampler pi0 needs (swap DDPM for the flow-matching Euler step).

## Phase 2 — resolve pi0's two unknowns (cheap probes, first pi0 disk spent here)
4. `bash scripts/setup_vla_export.sh` → venv-A (disk gate: needs ~6GB headroom).
   `.venv-lerobot/bin/python scripts/pi0_export_probe.py denoise --export-only` then
   `.venv/bin/python scripts/pi0_export_probe.py denoise --lower /tmp/pi0_probe`.
   - **GATE 2a (shape):** the probe prints `prefix_len`. 816 = 256 patch tokens/image;
     51 = pooled. **Freeze `PI0_PREFIX_LEN` to the resolved value before any full export.**
   - Confirms flow-matching + attention + block-mask ops lower.
5. `pi0_export_probe.py encode` (both venvs) → the one true unknown: SigLIP/PaliGemma on
   coremltools 9.0.
   - **GATE 2b (biggest risk):** if the vision tower won't lower, take the fallback —
     export ONLY `denoise_step` to `.aimodel`, keep the VLM prefix on MPS/torch and inject
     the prefix KV as a tensor input (clean: the experts interact only through attention).
     Don't force a full-model export.

## Phase 3 — full pi0 export + Gate B
6. `PI0_PREFIX_LEN=<resolved> .venv-lerobot/bin/python models/pi0/export.py export --out build/pi0-base`
   (venv-A; applies the :918 deepcopy strip + eager attn).
7. `.venv/bin/python models/pi0/export.py --lower --out build/pi0-base` (venv-B) → the two
   `.aimodel`s. Then write `metadata.json` (`num_steps=10`) + `norm_stats.json` in the recipe layout.
8. `run_action_parity` — reference precompute (venv-A, `--reference-cache <f>.npz`) then
   load-compare (venv-B). Recorded frames from a pi0-compatible LeRobot dataset, fixed noise seed.
   - **GATE 3:** `min_action_cosine ≥ 0.998` AND `max_normalized_mae ≤ 0.05`. If it reports
     `not_run`, the exported `denoise_step` didn't surface an injectable `x_t` — fix the
     wrapper's input contract, not the runner. **fabric never fakes a number.**
9. Publish (`coreai-fabric publish pi0-base`) — the action card + honest "needs a matching
   robot to actuate" banner already render (bundle_kind: action). Register into the catalog.

## Standing disk rule
This machine runs ~95% full. Phases 0–1 cost nothing pi0-specific; the first pi0 byte is
only written at step 4. Never start a phase whose peak disk exceeds current headroom —
free space (or use a roomier machine) first. The greedy_parity/action runs + the 14GB pi0
download are the disk-heavy steps.
