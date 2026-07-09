# Verification report — surgical registry RFC (post-implementation audit)

> **RESOLUTION 2026-07-09 — all audit defects fixed (197 tests pass, validate 0 errors, cross-contract green vs LIVE catalog).**
>
> | Defect | Fix | Evidence |
> |---|---|---|
> | **H1** (F6) numbers trapped in gitignored `build/` | `verify` now writes a durable `gate_b.measured` block into the recipe; `register._load_parity_report` reconstructs the report from it on a fresh clone; scorecard reads it. Backfilled the 2 measured recipes (fastwam-libero, lingbot-vla-v2) from local reports. | scorecard + blocks-index `--check` reproduce with **no build/**; `test_durable_measurement.py` |
> | **H2** (F7a) gate-flip guard blind to column-0 `status:` | regex `\s+`→`\s*`; test fixtures made faithful (column-0). | live proof fires on real diff; `test_gate_flip.py` (incl. `test_column0_status_flip_is_detected`) |
> | **H3** (F1) bytes-vs-params tripwire didn't fire on realistic config | dtype-aware bytes/param, `implied_params` uses intermediate_size+GQA+lm_head, threshold 3.0→1.5; golden re-fixtured with the REAL SenseNova stub + a realistic hidden-MoT. | normal 1.00x / hidden-MoT 1.92x; `test_analyze_golden.py::test_hidden_mot_fires_weight_bytes_tripwire` |
> | **M1** (F2) protocol not required-when-measured | `verify` always fills the 4 core protocol fields; `recipes.validate` errors on a `measured` recipe missing them (legacy exempt). | `test_durable_measurement.py` (M1 cases) |
> | **M2** (F7c) waivers didn't reach catalog | `register` folds waivers into the accepted `reason` field (no schema break). | `test_register_eval.py::test_waivers_surface_in_catalog_reason`; cross-contract green |
> | **L1/L3/L4/L5** | SDPA-mask smoke added; `--check` made freshness-stamp-insensitive + wired into CI + tests de-tautologized; dead code removed; explicit index-only exclusion guard. | `test_smokes.py`; `.github/workflows/validate.yml` |
>
> The original audit findings are preserved below for the record.

- **Date:** 2026-07-09
- **Method:** 5 parallel auditors (Phase 0, Phase 1, Phases 2+3, Phase 4+hygiene, adversarial re-attack of fatal findings), each running real commands + reading real code, against the working tree at HEAD `e54928e` and the LIVE catalog clone.
- **Global gates:** `pytest -q` → **180 passed**; `coreai-fabric validate` → **0 errors, 8 (legit) warnings**; 7 clean commits, one per phase.
- **Verdict:** mechanisms are substantially real and mostly well-built — but **the two FATAL integrity guarantees the redesign existed for are hollow at the decisive point**, and one flagship tripwire is cosmetically satisfied. 3 HIGH defects must be fixed before this is truly "done".

## HIGH — each reopens a fatal/flagship finding

### H1 (reopens F6) — measured Gate-B numbers are trapped in gitignored `build/`
`verify.py` writes the protocol *signature* back into the recipe, but **not the measured `value`**. The number lives only in `build/<id>/parity-report.json`, and `build/` is gitignored. On a fresh clone / CI (no local `build/`):
- `_catalog_evaluation` returns `None` → the registered catalog entry gets **no evaluation, no number** — the exact F6 defect the RFC set out to close.
- `docs/scorecard.md` value rows require `value` → **0 value rows regenerate on a clean clone** (committed file shows 3). Reproducible **only** on the author's machine.
- `fastwam-libero` still stamps `high_fidelity` via the `int8` quantization fallback with no reachable measurement (softly reopens F7 rule 2).
**Fix:** persist the measured `value` (+ status) into the recipe `gate_b` (durable in git), or commit parity reports outside `build/`. Then numbers reach the catalog and the scorecard reproduces off-author-machine.

### H2 (reopens F7 rule 1) — the same-commit gate-flip guard is blind to real recipe layout
`scripts/check_gate_flip.py:42` `_INDENT_FIELD_RE = r"^\+?\s+(threshold|metric|tolerance|status):"` requires ≥1 space after `+`. Every real recipe writes `status:` at **column 0**. Proven live: a diff that relaxes `threshold: 0.999→0.90` AND flips `+status: verified` → `(gate=True, status_flipped=False)` → **guard passes**. The pi0fast attack it exists to stop sails through. `tests/test_gate_flip.py` only uses *indented* `+    status:` fixtures — self-confirming test over an unfaithful layout.
**Fix:** match top-level `status:` (`\s*` for the status alternative, or a separate column-0 pattern); re-fixture the test with column-0 `status:`.

### H3 (reopens F1) — the bytes-vs-params tripwire does NOT fire on the real SenseNova config
`analyze.py` `WEIGHT_PARAMS_RATIO_TRIPWIRE = 3.0`. Against a realistic textbook Qwen2-7B config (hidden 3584, 28 layers, vocab 152064) vs the real 29.2GB checkpoint: config implies ~4.9B, bytes imply ~11.7B → **ratio 2.4x < 3.0x → verdict CANDIDATE, not MANUAL**. The golden test only "passes" the SenseNova case because its fixture uses an artificially shrunk 2-layer config (ratio 13.9x). The exact real-world catch the RFC advertises (§7.1.3) would leak as a confident CANDIDATE. Also `implied_params` undercounts a real 7B (~4.9B for ~7.6B).
**Fix:** lower the threshold (a 7B→MoT doubling is ~2x, not 3x), fix `implied_params` undercount, and re-fixture the golden test with the true 28-layer config.

## MEDIUM

- **M1 (F2 not enforced):** `gate_b.protocol` is fully optional in the schema (`gate_b.required` = metric/threshold/tolerance; no `if/then`/`dependentRequired`), and neither `validate_recipe` nor `register` rejects a measured recipe missing it. RFC §10 promises both. Consequence observed live: `lingbot-video-dense-1.3b` ranks in a `input_protocol: ?` scorecard cell — the incommensurable-cell leak F2 meant to close.
- **M2 (F7c):** waivers surface in the scorecard + demote fidelity tier, but do **not** reach the catalog `evaluation` (deferred with the protocol block to a batched catalog PR). RFC §4.4 rule 3 asserts they reach the catalog. As-is the catalog can carry a waivered pass with no waiver visible.
- **M3 (F8):** `run` + `error_signatures` are real and tested (0x10004→blocked, novel→unclassified/failed, exit-127 captured), but `attempts/` has **zero committed real records** (only README). Expected until an on-toolchain conversion runs, but the "committed structured data" substrate `/reflect` mines is empty.

## LOW

- **L1 (F16):** RFC §5.3 names 3 smokes; only 2 exist — the **SDPA-mask-forms smoke is missing**.
- **L2 (F3):** golden test covers **7 representative fixtures, not the 53** upstreams §7.2 promises.
- **L3 (F15):** the `--check` reproducibility guard for scorecard/blocks-index is a pytest but **not wired into any CI workflow**; the reproducibility tests overwrite-then-check (prove determinism, not committed-vs-clean).
- **L4:** dead code `register.py:134-151` (unreachable after `return ev`); unused `check_gate_flip.py:41 _FIELD_RE`.
- **L5 (F10):** index-only battery exclusion is design-implicit (smokes key to techniques not recipes); no explicit guard/test.

## Genuinely fixed (verified green)

- **F1 analyze refusal-first vocabulary** — only "candidate lane" / "MANUAL ANALYSIS REQUIRED"; no "SOLVED"/coverage-% anywhere in generated output. Tripwire *scaffolding* real (the threshold is the H3 problem).
- **F4** — `techniques.yaml` killed; playbook has T0–T7 + trigger index + framework scope + preconditions (graph-split bounded-host-loop; bool-mask lerobot-artifact, verbatim); `/reflect` ritual documented with oversight OUTSIDE the loop.
- **F5/F14** — optional `blocks:` array; single vocab authority `blocks-vocab.yaml`; unknown id → warning (tested); reverse index generated, `used_by` derived never stored, body reproduces byte-identically (modulo the freshness stamp).
- **F7 rule 2** — `_fidelity_tier` from (value−threshold)+n_obs, no pass→high_fidelity collapse (correct *when a build report exists*).
- **F10** — publish bundle content allowlist (default-deny) rejects stray `*.safetensors`, tested green; `.gitignore` weight guard present.
- **F11** — `loaded_on_ane` deployability facet distinct from cosine.
- **F13** — fictional fixture reconciled to live catalog vocab; traits enum reconciled; cross-contract check **exit 0 against the LIVE catalog**, wired into CI.

## Not RFC work — uncommitted in the tree

`models/lingbotvideo/`, `models/lingbotworld/`, `recipes/lingbot-video-*.yaml`, `templates/model-card-video.md`, and modified `publish.py` (+`_render_video_card`) / `scorecard.md` / `blocks-index.md` are the **LingBot-World V2 video lane** — a separate feature. ⚠️ The uncommitted `scorecard.md`/`blocks-index.md` carry a `toolchain_version` **stub downgrade** (`torch==2.11.0`) from regeneration on an incomplete local toolchain — do not commit that stamp as-is.
