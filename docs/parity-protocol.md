# Gate B — numeric parity protocol

Status: **protocol defined; a conforming runner ships with fabric.**
`coreai-fabric-parity-runner` (installed by the `[convert]` extra) implements
`per_token_logit_cosine` in Python on the Core AI runtime that ships inside
the `coreai-core` PyPI wheel — verified on real hardware (Apple Silicon,
macOS 26, 2026-07-03): the runtime loads, specializes and executes
`.aimodel` assets, so Gate B needs neither a Swift runner nor macOS 27.
`graph_output_cosine` still needs a runner (per-modality preprocessing);
fabric's runner exits non-zero for it rather than faking a result. Fabric's
`verify` shells to whatever `COREAI_FABRIC_PARITY_RUNNER` names and honestly
records `not_run` when none is configured.

**Which metric applies depends on the EXPORT LAYOUT, not just the modality.**
The logit/graph-cosine metrics compare a STATIC-graph export (fabric's
`coreai-fabric-llm-export`: fixed `(1, seq_len)` `input_ids` → full-sequence
`logits`, `use_cache=False`) against an fp32 reference. A PRODUCTION
`coreai.llm.export` asset is different: it is a **quantized (e.g. 4bit),
stateful KV-cache** asset whose raw logits legitimately diverge from fp32 —
raw cosine is the wrong metric for it. Its Gate B is `benchmark_accuracy`
(below), which is blocked upstream. See `docs/validation-log.md`
(2026-07-03 production run) for the measured descriptor
(`state_names=['keyCache','valueCache']`, dynamic `[1,-1]` inputs) that makes
this concrete.

## Purpose

Gate A proves the bundle is structurally sound. Gate B proves the converted
`.aimodel` computes (numerically) the same function as the upstream model it
claims to be. Thresholds follow the convention established by community
porting practice: cosine similarity ≥ 0.999, plus greedy-token-exact decoding
for LLMs.

## Metrics (recipe `parity.gate_b.metric`)

### `greedy_parity` (the runnable Gate B for production LLM assets)
For a PRODUCTION stateful `coreai.llm.export` asset — the metric the community
reports ("X/Y token-exact vs fp32"), and the one fabric can actually **run**
(unlike `benchmark_accuracy`, which is blocked on Apple's stubbed evaluator):
1. Fix a deterministic prompt set (seeded).
2. For each prompt, compute the fp32 reference's greedy continuation (K tokens)
   — the teacher path.
3. Drive the asset's REAL KV-cache decode (`coreai-fabric-parity-runner`,
   validated on macOS 26 / Apple Silicon) and, teacher-forced along the reference
   tokens, record whether the asset's argmax equals the reference's next token
   at each step.
4. Report `value` = the fraction of per-token argmax agreements (`matched` /
   `compared`), plus `greedy_token_exact` (all matched) and a decoded `sample`.

Contract (general across LLM assets; all dims read from the descriptor):
`input_ids` [1,seq] + `position_ids` [1,seq] (int32) → `logits` [1,seq,vocab],
with `keyCache`/`valueCache` states [n_layers,1,n_kv_heads,seq,head_dim]. The
decode contract (coreai-models `qwen3.py:86`, `offset = seq_len - query_len`):
`input_ids` is the NEW token; `position_ids` is the FULL range `[0..pos]` (its
length is the total sequence length). If an asset does not match this contract,
the runner reports `not_run` — never a fake number.

**Cost:** the coreai-core Python reference runtime is correct but SLOW
(~0.16 tok/s on Apple Silicon), so `greedy_parity` is an **opt-in local** check (via
`COREAI_FABRIC_PARITY_RUNNER`), not a fast/CI gate. Real tok/s throughput needs
the on-device Swift runtime (macOS 27).

#### Fair-measurement defaults (grounded in a 2026-07-03 root-cause study)

A raw greedy-argmax-exact number is misleadingly low unless the experiment is
fair. These are the DEFAULTS, so numbers are trustworthy and only compared
like-for-like:

- **Reference precision = the asset's own compute precision (float16)**, not
  fp32. The export computes at fp16; referencing fp32 blames the fp32→fp16
  rounding on the quantizer. `--reference-dtype float32` gives the stricter "vs
  fp32 oracle" number (what community cards quote) as a labeled secondary.
- **Margin-gated (near-tie budget):** a reference argmax disagreement counts as
  a real flip only when the reference top1−top2 margin exceeds `--flip-margin`
  (0.1 nats). At a near-tie the fp16/fp32 reference itself flips on noise;
  counting that as a quant failure is misleading. The **primary `value` is the
  margin-gated rate**; raw `argmax_match_rate`, `top5_agreement_rate`, and a
  **Wilson 95% CI** are reported alongside as diagnostics.
- **Sample floor N≥8 prompts.** A small window cannot claim high fidelity — the
  CI makes the uncertainty explicit.
- **Every number carries its precision signature** `(reference_dtype,
  flip_margin, n_prompts, decode_len)`. Compare int8-vs-int8 and int4-vs-int4,
  never across tiers.

#### Quantization tier and the "lossless" rule

Root-cause finding (verified against the community's own measurements): **int4
does not survive on Qwen-family models** — the community ships **int8** for a
near-lossless claim and has directly measured that int4 (k-means g32) fails
their oracle gate, while Apple's macOS int4 preset additionally uses
`symmetric_with_clipping`, which flips argmaxes on the fat-tailed LM head.
Therefore:

- **int4** is the size-optimized *shipping* tier (Apple's macOS default). Its
  greedy_parity is measured and reported, but it is **never labeled "lossless."**
- **"lossless" / near-lossless parity** may only be attached to an **int8**
  (weight-only, per-block-32 or per-channel, symmetric absmax, **no clipping**;
  attention/RoPE/RMSNorm kept high-precision) artifact with an adequate sample.
- The certifying `benchmark_accuracy` gate stays `not_run` until Apple's
  `coreai.llm.eval` ships — for everyone.

### `graph_output_cosine`
For non-autoregressive models (vision, audio encoders, embeddings):
1. Fix a deterministic input set (seeded; N ≥ 8 inputs appropriate to the
   input modality).
2. Run the upstream model (reference implementation, e.g. PyTorch at the
   pinned `upstream.revision`) and the `.aimodel` bundle on identical inputs.
3. Flatten each pair of output tensors; compute cosine similarity per input.
4. Report the MINIMUM cosine across inputs as `value`.

### `per_token_logit_cosine`
For autoregressive LLMs:
1. Fix a deterministic prompt set (seeded; N ≥ 8 prompts) and a decode length
   (default 64 tokens, greedy).
2. At each decode step, compute cosine similarity between the upstream and
   converted next-token logit vectors.
3. Report the MINIMUM per-token cosine across all steps and prompts as
   `value`.
4. If the recipe sets `greedy_token_exact: true`, additionally report whether
   the greedy token sequences match exactly (`greedy_token_exact: true/false`).

### `benchmark_accuracy`
For a PRODUCTION `coreai.llm.export` asset (quantized, stateful KV-cache): the
correct fidelity metric is **task accuracy vs upstream**, not raw logit
fidelity. A 4bit asset is expected to diverge in logits while preserving
downstream accuracy, so the gate is: the converted asset stays within
`tolerance` of the upstream model's score on a fixed benchmark (e.g. tinyMMLU
/ tinyGSM8k), evaluated through the KV-cache decode path.

**Status: blocked UPSTREAM (not_run).** The only conforming evaluator is
Apple's `coreai.llm.eval` (KV-cache-aware), which ships as a STUB in
coreai-models 0.1.0 — it prints "Evaluation support is coming soon". A
static-graph runner cannot score a stateful asset. So `verify` reports
`not_run` with that reason for any recipe whose metric is `benchmark_accuracy`
— regardless of whether a runner is configured — and never records it as a
failure. When Apple ships the evaluator, the runner contract below gains a
`benchmark_accuracy` implementation that shells to `coreai.llm.eval`.

### `action_parity` (the runnable, robot-free Gate B for a VLA/robot policy)
A robot policy (VLA) is not an LLM: it maps `(images, proprioceptive state,
language instruction)` → a **continuous action chunk** `[chunk_size, action_dim]`,
via an iterative flow-matching sampler (a fixed `num_steps`, e.g. 10). There is
no vocabulary and no argmax, so `greedy_parity` does not apply. `action_parity`
is the structural analog:

- **Fair by construction — fix the noise.** The flow-matching sampler is
  deterministic *given its initial noise*. `action_parity` **injects a fixed
  seed noise** on both sides, exactly as `greedy_parity` teacher-forces the
  reference's greedy path and never calls `generate`. This turns the sampler
  into a deterministic function of `(observation, noise)`, making a numeric
  comparison honest.
- **Inputs are REAL, never invented.** Recorded `(images, state, instruction)`
  from a published LeRobot dataset (flagship `lerobot/svla_so101_pickplace`);
  the instruction is the dataset's real recorded `task` string.
- **Reference:** the upstream policy in torch at `--reference-dtype` (default
  `float16` to isolate quantization error; `float32` for the stricter oracle),
  same fixed noise, same `num_steps`.
- **Numbers (per sample, aggregated over N≥8 seeded frames across ≥2 episodes),
  compared in NORMALIZED action space:** primary `value` = per-sample chunk
  cosine reported as the **minimum** across samples (one worst frame can't hide
  behind an average, matching the existing cosine metrics); diagnostics recorded
  verbatim: `mean_action_cosine`, `max_normalized_mae`, `mean_normalized_mae`,
  `per_dim_mae`, `first_action_mae` (the action actually executed under
  receding-horizon control), and a **bootstrap 95% CI** on mean cosine (the
  continuous-metric analog of `greedy_parity`'s Wilson CI).
- **Precision + step signature.** The report tags `(num_steps, reference_dtype,
  n_obs, chunk_len)` so tiers only compare like-for-like — the same rule as the
  greedy_parity precision signature. A 10-step and a 4-step export are DIFFERENT
  models with separately-measured fidelity; a reduced-step tier never cites the
  10-step number.
- **`not_run` (unmeasured, not a failure)** when the asset doesn't expose the
  deterministic-noise sampler contract (no injectable noise / internal un-seeded
  RNG), or when the preprocessing stats (`meta/stats.json` resize-pad + MEAN_STD)
  don't hash-match on both sides — a silent stats mismatch would yield a
  meaningless number. Mirrors the `greedy_parity` not_run discipline exactly.

**What it PROVES:** the export computes the SAME function as the source policy —
catching quantization drift, wrong normalization constants, image-preprocessing
mismatch, a broken denoise loop, layer-fusion bugs. **What it does NOT prove
(mandatory card label):** real-world task success, embodiment transfer, or
closed-loop stability. The closed-loop success-rate eval (LIBERO/ManiSkill) is a
separate future gate, `not_run` here — the VLA analog of `benchmark_accuracy`
being blocked. Never conflate the two.

## Pass criteria

```
pass  ⇔  value ≥ threshold − tolerance
         AND (greedy_token_exact if required by the recipe)
```

`threshold` and `tolerance` come from the recipe — the runner must not
hardcode them.

## Runner contract

`coreai-fabric verify` invokes the configured runner
(`COREAI_FABRIC_PARITY_RUNNER`) as:

```
<runner> --bundle <path/to/id.aimodel> --upstream <owner/name> \
         --metric <metric> --threshold <t> --tolerance <tol> --report-json - \
         [--revision <sha>]
```

`--revision` is appended when the recipe pins `upstream.revision`; runners
SHOULD compare against exactly that revision (fabric's runner does).

and expects a single JSON object on stdout:

```json
{
  "value": 0.9994,
  "greedy_token_exact": true,
  "n_inputs": 8,
  "runner": "name/version",
  "environment": {"os": "...", "chip": "...", "runtime_version": "..."}
}
```

Only `value` is required for the pass computation; everything else is
recorded verbatim into `parity-report.json`. A non-zero exit or non-JSON
stdout is recorded as a Gate B failure with the stderr excerpt.

## What remains TODO (honest gaps)

- `benchmark_accuracy` for production `coreai.llm.export` assets is blocked on
  Apple shipping `coreai.llm.eval` (a stub in coreai-models 0.1.0). Until then,
  a production recipe's Gate B is `not_run` by design — Gate A validates the
  bundle structure, and quality parity waits for the upstream evaluator. This
  is the single largest gap between "conversion works" and "conversion
  verified" for the production path.
- `graph_output_cosine` support in fabric's runner (per-modality reference
  preprocessing; input corpora for audio/images). The LLM prompt corpus is
  versioned in `coreai_fabric/parity_runner.py` (`PROMPTS`).
- A Swift runner on the OS Core AI runtime (macOS/iOS 27) to additionally
  prove parity on the DEPLOYMENT runtime — fabric's runner exercises the
  runtime bundled in the coreai-core wheel, which is the same stack but not
  the on-device binary.
- A decision on preprocessing parity for models with `processor_required`
  (the processor must be identical on both sides or the comparison is
  meaningless).
