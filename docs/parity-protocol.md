# Gate B — numeric parity protocol

Status: **protocol defined; a conforming runner ships with fabric.**
`coreai-fabric-parity-runner` (installed by the `[convert]` extra) implements
`per_token_logit_cosine` in Python on the Core AI runtime that ships inside
the `coreai-core` PyPI wheel — verified on real hardware (Apple M4 Max,
macOS 26.6, 2026-07-03): the runtime loads, specializes and executes
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
   validated on macOS 26.6 / M4 Max) and, teacher-forced along the reference
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
(~0.16 tok/s on M4 Max), so `greedy_parity` is an **opt-in local** check (via
`COREAI_FABRIC_PARITY_RUNNER`), not a fast/CI gate. Real tok/s throughput needs
the on-device Swift runtime (macOS 27).

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
