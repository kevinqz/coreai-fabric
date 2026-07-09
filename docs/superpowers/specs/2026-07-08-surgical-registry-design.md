# RFC — coreai-fabric surgical registry & harness loop

- **Status:** ✅ Implemented (fabric-side), 2026-07-09. All phases 0–4 + hygiene landed; every confirmed finding F1–F16 traced to an artifact (Appendix A). The catalog-side `evaluation` schema PR (accept `value`/`min_cosine`/`protocol`) remains a documented batched follow-up to coreai-catalog — fabric emits the richer block via `register.catalog_protocol_extension()` and the cross-contract check stays green against the live catalog.
- **Date:** 2026-07-08
- **Author:** kevinqz + Claude Code session
- **Supersedes:** the "block+technique registry" proposal (killed/redesigned by the 2026-07-08 redteam)
- **Redteam record:** 28-agent adversarial workflow, 40 raw → 18 canonical → 17 confirmed / 1 refuted findings. Full report archived at `docs/superpowers/specs/redteam-2026-07-08.md` (see Appendix B).

---

## 1. Motivation

A new model (SenseNova-Vision-7B-MoT) was added this session as an index-only draft. Decomposing it by hand revealed that most of its blocks (SigLIP encoder, Qwen2 LM forward, MoT-split) overlap with work already done for other models. The original ambition — a normalized `blocks.yaml` registry + `techniques.yaml` + an `analyze` auto-decomposer + a per-block "best-lane SotA" ranking + self-improvement loops — aimed to capture that overlap automatically and keep everything "SotA".

An intense redteam (grounded in the real repos) **falsified the grand version on every axis** and reduced it to a surgical package. The headline reasons:

- **There are no per-block numbers to register.** Every Gate B scores the *composed* deployable graph. Attributing a whole-graph cosine to a constituent block is attribution fraud (F1, F6).
- **"The Gate-B number" is four incommensurable quantities** under unrecorded protocols; ranking over it crowns the *weakest* verification (F2, F7).
- **Config-reading decomposition works on ~0 of the multi-block upstreams** it targets; the flagship's own configs conceal its MoT (F1, F3).
- **Auto-applying techniques has no execution vehicle** — techniques are trace-time source edits in hand-vendored torch, and fabric refuses by design to drive those scripts (F4).
- **The self-improvement loop has no data substrate** — failed exports leave zero structured trace, and fabric never sees 36/53 script-tool runs (F8).

This RFC specifies what survived: a **protocol-honest evaluation record**, a **failure substrate**, an **optional block tag with a derived index**, an **honest reduced `analyze`**, and a **playbook-as-ACE reflect loop**. Every design decision below is traceable to a confirmed finding (§9 matrix).

## 2. Goals / Non-goals

**Goals**

1. Make every Gate-B measurement carry its full protocol signature, and make the numbers reach the catalog durably.
2. Capture every conversion attempt — **including failures** — as committed structured data.
3. Let a recipe *optionally* declare its architectural blocks, and derive a reverse index (block → recipes × envelope × measured Gate-B) by generation.
4. Give `analyze <hf_repo>` an honest, refusal-first decomposition that never fabricates a "SOLVED" or a coverage %.
5. Close a bounded reflect loop: mine attempts → propose bounded playbook/vocab diffs → validate against a cheap smoke battery → human-reviewed commit.

**Non-goals (explicitly out of scope, per redteam)**

- A standalone block *entity* in the catalog (new 7th contribute/validate/audit/counts/generate surface). — F5, F14
- Per-block Gate-B numbers or a per-block leaderboard. — F1, F6
- `techniques.yaml` / auto-applied techniques. — F4
- A "coverage %" or "SOLVED" vocabulary anywhere. — F1, F3
- Cross-metric-family "best lane" ranking. — F2
- Any scheduled/autonomous acceptance of registry changes without human review. — Weng oversight

## 3. Architecture overview

```
                 ┌─────────────────────────── fabric repo ────────────────────────────┐
   hf_repo ──▶ analyze ──▶ recipe.yaml ──▶ run ──▶ verify ──▶ publish ──▶ register ──▶ catalog
               (F3)          │  blocks:[] │  (F8)   │ gate_b   │ allowlist │ _EVAL_KEYS   (derived
               refusal-first │  (F5)      │ attempts│ .protocol│ (F10)     │ + protocol   facet)
               weight-bytes  │            │ /*.jsonl│ (F2)     │           │ (F2/F6)
               tripwire (F1) │            ▼         ▼          │           │
                             │        attempts/ (committed, incl. failures)│
                             │            │                                │
                             ▼            ▼                                ▼
                    docs/blocks-index.md  smokes/ (proxy battery, F16)   docs/scorecard.md
                    (GENERATED, F14)      │                              (GENERATED, F6 §7)
                             ▲            ▼
                             └──── /reflect ritual: mine attempts → bounded playbook/vocab diffs
                                   → run smokes → human-reviewed commit (Weng: oversight OUTSIDE loop)
```

Everything canonical lives **fabric-side** (single vocabulary authority, F13). The catalog receives a *batched derived facet*; it never becomes a second source of truth.

## 4. Phase 0 — Protocol-honest evaluation record *(F2, F6, F7; prerequisite for everything)*

### 4.1 `gate_b.protocol` schema extension

`schema/recipe.schema.json` — the single `gate_b` object gains a required-when-measured `protocol` object:

```yaml
gate_b:
  metric: graph_output_cosine        # existing; the metric FAMILY
  threshold: 0.999
  tolerance: 0.0005
  protocol:                          # NEW — the measurement signature
    n_obs: 8
    seed: 0
    input_protocol: synthetic        # synthetic | recorded | mixed
    reference_dtype: float32         # float32 | float16 | bfloat16
    granularity: flattened           # per_row | flattened | per_token
    graph_boundary: "encode→denoise" # what the number spans
    loaded_on_ane: true              # deployability signal, distinct from parity (F11)
    waivers: []                      # e.g. ["near_zero_action"]; surfaced, never silent
```

`input_protocol` replaces the schema description that **falsely claims recorded LeRobot inputs** — the field records the truth (many drivers measure on `torch.rand`, with the smolvla 0.02-cosine catastrophe as the standing warning).

### 4.2 verify writes the protocol

`coreai_fabric/verify.py` already computes all of these — they currently die in stdout and gitignored `build/<id>/parity-report.json`. Change: `verify` writes `gate_b.protocol` back into the recipe (or a committed sidecar `evaluations/<id>.yaml` — see §11 Q3) on every run, **pass or fail**.

### 4.3 register carries numbers to the catalog

`coreai_fabric/register.py::_EVAL_KEYS` — extend the whitelist to include `value, min_cosine, median_cosine, n_obs, threshold` and the full `protocol` block. Today `_EVAL_KEYS` **drops the numeric value**, so the live catalog holds no Gate-B number for any action lane. This is a ~1-line-class fix plus a one-time batched catalog PR that extends the `evaluation` schema to accept the fields.

### 4.4 Integrity rules (the anti-gaming gates, F7)

1. **No same-commit gate flip:** a CI check rejects a diff that changes a `gate_b.threshold`/`metric`/`tolerance` **and** flips any recipe's status to a passing state in the same commit. (pi0fast precedent: relaxations landed with the results they enabled.)
2. **Fidelity tier from margin, not from pass:** `register.py::_fidelity_tier` must derive the tier from (value − threshold) and `n_obs`, never collapse *any* pass to `high_fidelity`.
3. **Waivers surface:** the `near_zero_action` (and any) waiver appears in the catalog `evaluation`, never silently marks passed below the bar.

### 4.5 SotA table is generated, per-cell only

`docs/scorecard.md` is **generated** from recipes (playbook §7 is the hand-built prototype). Rankings exist **only within a `(metric, granularity, input_protocol, reference_dtype, graph_boundary)` cell**, on raw values, with waiver flags shown. No cross-cell "best". This *is* Layer D, honestly reduced.

## 5. Phase 1 — Failure substrate: `fabric run` + `attempts/` *(F8, F16)*

### 5.1 `coreai-fabric run <id>`

New CLI subcommand. Today `convert` *prints* the venv-aware invocation and refuses to drive script tools by design; `run` executes that exact printed invocation and instruments it. It is the single choke point the loop needs — **the drivers do not change**.

Captured per attempt → appended to `attempts/<id>.jsonl` (**committed**):

```json
{"ts":"2026-07-08T20:10:00Z","recipe":"sensenova-vision-7b-mot","stage":"convert",
 "tool":"models/sensenova/export.py","exit":1,"toolchain":"coreai_torch==0.4.1",
 "error_signature":"0x10004","error_tail":"...appleneuralengine Program load failure (0x10004)...",
 "envelope":{"precision":"fp16","layers":18},"outcome":"blocked"}
```

`error_signature` is a **distilled class**, matched by a small ordered regex table (`coreai_fabric/error_signatures.py`): `0x10004`, `complex128`, `FoldMultiplyIntoSDPAScale`, `versioned_IR`, `OOM`, `import_error`, `parity_below_threshold`, … → falls back to `unclassified` (which is itself a signal the table needs a new entry — a loop input).

### 5.2 Status + verify-on-failure

- New recipe statuses: `blocked` (ANE ceiling / license / toolchain-skew) and `failed` (attempted, did not pass). `NEXT_STEP` gains entries for both.
- `verify` writes its result on **failure** too (today it writes status only on pass).

### 5.3 Smoke battery `smokes/` (F16)

Promote the existing scratchpad random-weight smokes (MoE dense-fusion lowering, graph-split chaining, SDPA mask forms) to a committed `smokes/` dir, each keyed to the technique/block it exercises. This is the loop's **cheap local evaluator** — no fleet re-verification. A header in each smoke **explicitly excludes scale-dependent effects** it cannot see (0x10004 program ceiling, fp16 load ceiling) so a green smoke is never mistaken for a loadability guarantee (F12).

## 6. Phase 2 — Optional `blocks:` + derived index *(F1, F5, F9, F12, F14)*

### 6.1 Recipe schema (~10-line diff)

```yaml
catalog:
  # ...existing...
  blocks:                    # NEW, OPTIONAL, array of vocab strings
    - siglip-so400m-vit
    - qwen2-7b-lm
    - flux-vae
```

The **recipe remains the provenance record** — `upstream.hf_repo/revision/license` already live there, so a block string carries **no** license/weights of its own (kills F9's "block under three licenses" problem: the license axis stays per-recipe). Unknown block id → `validate` **warning**, never error.

### 6.2 Vocabulary authority (F13)

`schema/blocks-vocab.yaml` (fabric-side, single authority) lists valid block ids + a one-line human description each. The catalog does **not** mirror this; if a block facet is ever wanted catalog-side it is a *generated derived facet*, batched. Before adding it, reconcile the existing `traits` enum drift and fix/delete the fictional catalog test fixture (F13 hygiene, §8).

### 6.3 Derived index — **generated, never stored** (F14)

`scripts/generate_blocks_index.py` → `docs/blocks-index.md`: for each block, a table of `recipe × envelope × measured-Gate-B (from Phase 0) × status`. Because it is generated on demand from recipes, two in-flight registrations of SigLIP-bearing models **never collide on shared lines** — the PR #25–28 merge-collision class does not reappear. `used_by` is *never* a stored field.

### 6.4 Honest status vocabulary (F1, F12)

The index reports **`measured @ <envelope>`** (e.g. `qwen2-7b-lm: measured @ {non-AR denoise, fixed-shape, int8} via eo1-3b`), never `SOLVED`. A block "known" at one envelope tuple `(graph, static shape, precision, host-contract)` carries **no** promise at another — SenseNova's SigLIP-so400m @ 980px/patch-14 inside a MoT is `NET-NEW`, not covered by lingbot's ViT-B/16 @ 512px. Loadability is **never** inferred from block facts (F12: same MoE stack loads at L=12 fp16, fails at L=18).

## 7. Phase 3 — `analyze <hf_repo>`, reduced and refusal-first *(F1, F3)*

`coreai-fabric analyze <hf_repo>` — an upgrade of `new`'s metadata read. It **never** emits "SOLVED" or a coverage %. Output vocabulary is exactly **"candidate lane — verify against modeling code"** or **"MANUAL ANALYSIS REQUIRED"**.

### 7.1 What it does

1. **LeRobot parser:** `config.json type` → driver family, with **verbatim** shape prefill (it does *not* invent shapes; pi0's own driver says shape is "UNCERTAIN until probed").
2. **transformers exact-match:** on `architectures[0]` **only within a proven size/shape envelope** derived from converted recipes.
3. **Refusal tripwires (hard):** any of `auto_map`/`trust_remote_code`, stub config, gated repo, `model.pt`-only, or **weight-bytes vs config-implied-params contradiction** → forces `MANUAL ANALYSIS REQUIRED`. The bytes-vs-params cross-check is the tripwire that would have caught SenseNova's hidden MoT (29.2GB checkpoint vs a textbook-Qwen2 stub config).
4. **Prediction logging:** every candidate-lane emission is written to `attempts/` and later linked to the eventual Gate-B outcome, so **matcher precision becomes measurable** (today it is unknowable and can never improve).

### 7.2 Golden retrodiction test

`tests/test_analyze_golden.py`: run `analyze` against the 53 known upstreams; for each it must **either refuse or match correctly**. A wrong confident match fails the test. This gives a precision baseline for free and guards regressions.

## 8. Phase 4 — Playbook-as-ACE + `/reflect` ritual *(kills B; F4, F11; the loop)*

**`techniques.yaml` does not exist.** The playbook `docs/coreai-conversion-playbook.md` stays the canonical prose (it already *is* an ACE playbook: numbered, frequency-ordered, trigger-phrased). It gains only:

- **Technique ids** (`T1…Tn`) and a trigger-phrase index at the top.
- **Framework scope + explicit preconditions** per technique (F11): e.g. graph-split is valid *only under a bounded, small host-loop pass count*; the bool-mask monkeypatch is a **lerobot artifact** (triplicated in exactly the three lerobot lanes, absent elsewhere). A **deployability facet** (`loaded_on_ane`, from §4.1) is tracked distinct from cosine parity — an AR generator that reloads N multi-GB programs per token can pass cosine while being undeployable.

**The `/reflect` loop (end-of-session, human-gated — Weng: oversight OUTSIDE the loop):**

1. **Mine:** read new `attempts/*.jsonl` since last reflect; cluster by `error_signature` (verifier-grounded weakness mining).
2. **Propose (bounded):** diffs limited to the playbook, `blocks-vocab.yaml`, and `error_signatures.py` — never to arbitrary driver internals. Each proposal names the attempt records that justify it.
3. **Validate:** run the `smokes/` battery; a proposal that regresses a smoke is rejected.
4. **Human-reviewed commit:** the diff lands only via a commit the maintainer reviews. The loop **never** blocks on catalog PR merge (refuted finding: registration is already async).

Shipped as a repo skill/ritual doc so the step is explicit, not implicit discipline.

## 9. Cross-transversal hygiene *(F10, F13, F15)*

- **F10 — content gates on weights:** `publish.py` gains a **bundle content allowlist** (refuses stray `*.safetensors` / raw upstream slices in the bundle dir; `upload_folder` currently ships recursively with no allowlist). A `.gitignore` + CI guard rejects committed extracted block weights (derivative data of NC upstreams). Index-only recipes are **explicitly excluded** from any smoke/regression battery, never silently skipped.
- **F13 — vocabulary drift:** fix or delete the fabric test fixture that validates against a fictional catalog schema; reconcile the `traits` enum that empirically hard-fails catalog validation — **before** any new vocabulary (`blocks-vocab.yaml`) is added.
- **F15 — freshness:** every generated registry fact (`blocks-index.md`, `scorecard.md`) carries `verified_at` + `toolchain_version`; a CI invariant couples recipes to their block references so a rename can't silently orphan the index.

## 10. Error handling & testing

| Surface | Handling |
|---|---|
| `run` on a crashing driver | non-zero exit captured, `error_signature` distilled, `outcome:failed/blocked` written; never swallowed |
| `analyze` on hostile config | refusal-first; tripwire → `MANUAL ANALYSIS REQUIRED`; never a confident wrong match |
| unknown block id in recipe | `validate` warning, not error (YAGNI: don't block conversion on vocab lag) |
| missing `gate_b.protocol` on a measured recipe | `validate` error post-migration; catalog `register` rejects evaluations missing the signature |
| smoke regression during `/reflect` | proposal rejected, logged; no commit |
| `unclassified` error_signature | recorded as-is; flagged as a loop input to extend the signature table |

**Tests:** schema round-trip for `protocol` + `attempts` records; the F7 same-commit-gate-flip CI check; `analyze` golden retrodiction over 53 upstreams; the `smokes/` battery itself; a publish-allowlist test with a planted stray safetensors; a generation test that `blocks-index.md`/`scorecard.md` are reproducible from recipes (no drift).

## 11. Sequencing & dependencies

Phase 0 is a **hard prerequisite** (everything else references the protocol signature and the durable numbers). Then:

- **0 → 1:** `run`/`attempts` reuse the protocol writer.
- **1 → 3:** `analyze` prediction-logging writes into `attempts/`.
- **0 → 2:** the derived index reads Phase-0 numbers.
- **1,2 → 4:** `/reflect` mines `attempts/` and validates against `smokes/`; proposes into playbook + `blocks-vocab.yaml`.
- **Hygiene §9** runs alongside; F13 gates Phase 2's vocab add.

Recommended order: **0 → 1 → 3 → 2 → 4**, hygiene interleaved. Each phase is independently shippable and valuable; 0+1 alone deliver the integrity foundation the redteam says everything presupposes.

## 12. Resolved open questions (from redteam §4)

1. **Block identity key:** architecture id only, as an *optional recipe tag*; all envelope/precision/provenance nuance lives in the recipe + the generated index. We accept degeneration to per-recipe facts — that *is* the design (F5).
2. **Vocabulary authority:** fabric owns block ids and evaluation vocabulary; catalog is a batched derived facet. Fictional fixture + traits drift fixed first (F13).
3. **Protocol signature fields:** `n_obs, seed, input_protocol, reference_dtype, granularity, graph_boundary, loaded_on_ane, waivers` — catalog rejects measured evaluations missing them. (Home: `gate_b.protocol` in-recipe; §4.2 may move to `evaluations/<id>.yaml` sidecar if in-recipe churn is high.)
4. **Failure capture:** `coreai-fabric run` wrapper around the printed invocation (chosen over per-driver helper: one choke point, drivers untouched).
5. **License routing:** license stays per-recipe (blocks carry none); `analyze` per-block identifications feed `triage_license` *and* the publish gate; F10 allowlist keeps extracted slices out of bundles and git.
6. **Anchor namespace:** blocks proven only by draft/index-only recipes live in the **fabric recipe-id namespace** with the Phase-0 measured-parity field; no new catalog availability state (catalog rejects it today).

## 13. Out of scope / future

Catalog-side block entity; per-block cosine; technique auto-apply; coverage %; cross-cell SotA ranking; scheduled autonomous loop acceptance. Revisit only if fabric grows past a solo maintainer + agent-session scale (the redteam's churn baseline says the value/cost inverts well above N=53).

---

## Appendix A — Findings → design traceability

Each finding now maps to the **commit + artifact** that implements it (landed 2026-07-09):

| Finding | Sev | Addressed by (artifact) |
|---|---|---|
| F1 false-SOLVED generator | fatal | `analyze.py` refusal + bytes tripwire; `blocks-index.md` `measured @` vocab (never SOLVED); `test_analyze_golden.py` |
| F2 Gate-B = 4 quantities | fatal | `gate_b.protocol` schema; `verify.protocol_from_report`; `generate_scorecard.py` per-cell only |
| F3 analyze base-rate ~0 | fatal | `analyze.py` refusal-first + bytes tripwire; `test_analyze_golden.py` precision baseline |
| F4 no execution vehicle | fatal (KILL B) | no `techniques.yaml`; playbook-as-ACE (Tn ids); `reflect-ritual.md` |
| F5 blocks not reusable units | major | optional `catalog.blocks`; `blocks-vocab.yaml`; `generate_blocks_index.py` |
| F6 no per-block #, no durable home | major | `_catalog_evaluation` carries numbers; `catalog_protocol_extension`; `generate_scorecard.py` |
| F7 evaluator too weak/negotiable | major | `_fidelity_tier` from margin/n_obs/waivers; `check_gate_flip.py` CI; waivers surface |
| F8 no weakness-mining substrate | major | `error_signatures.py`; `run.py`; `attempts/*.jsonl`; `failed`/`blocked` statuses |
| F9 no license/provenance axis | major | block ids carry no license (stays per-recipe); analyze feeds `triage_license` |
| F10 no content gate on weights | major | `publish.assert_bundle_content` allowlist; `.gitignore` weight guard |
| F11 framework-keyed triggers / precondition | major | playbook framework scope + preconditions; `loaded_on_ane` deployability facet |
| F12 envelope-tuple facts / 0x10004 | major | `blocks-index` envelope keying; `smokes/` exclusion headers |
| F13 two-repo vocab contract broken | major | single fabric-side `blocks-vocab.yaml`; evaluation reconciled to LIVE catalog; cross-contract green |
| F14 stored used_by → merge collisions | major | `used_by` derived by generation, never stored |
| F15 freshness vs moving toolchain | minor | `verified_at` + `toolchain_version` on generated docs; orphan-ref flag |
| F16 zero-regression unclosable | minor | `smokes/` proxy battery with explicit scale-effect exclusions |
| (refuted) human-PR-merge-in-loop | — | reflect loop never blocks on catalog merge |

Commit trace (one per phase): `b283b92` Phase 0 · `d6c6bf3` Phase 1 · `c8120a7` Phase 3 · `c7418b8` F13 · `bf2ab2d` Phase 2 · `94a39d4` Phase 4 + F10/F15.

## Appendix B — Redteam provenance

28-agent workflow (`wf_1cf5ec07-6fd`): 8 lens critics (block-identity, maintenance-economics, automation-fragility, catalog-contract, sota-metric-integrity, harness-loop, license-provenance, scope-alternatives) → ACE-style dedup curator → one adversarial refuter per canonical finding → synthesis. Every confirmed finding cites file paths/lines verified against `coreai-fabric` and `coreai-catalog`. Layer verdicts: A NEEDS REDESIGN, B KILL, C NEEDS REDESIGN, D NEEDS REDESIGN, loops NEEDS REDESIGN. Full report: `docs/superpowers/specs/redteam-2026-07-08.md`.
