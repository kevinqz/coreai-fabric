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

## Purpose

Gate A proves the bundle is structurally sound. Gate B proves the converted
`.aimodel` computes (numerically) the same function as the upstream model it
claims to be. Thresholds follow the convention established by community
porting practice: cosine similarity ≥ 0.999, plus greedy-token-exact decoding
for LLMs.

## Metrics (recipe `parity.gate_b.metric`)

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
