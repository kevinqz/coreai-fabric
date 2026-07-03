# Gate B — numeric parity protocol

Status: **protocol defined, runner not yet implemented.** Gate B execution
requires macOS on Apple Silicon with the Apple Core AI runtime; fabric's
`verify` command shells to a runner and honestly records `not_run` when none
is configured. Nothing in this document has been measured yet — it specifies
what a conforming runner MUST do.

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
         --metric <metric> --threshold <t> --tolerance <tol> --report-json -
```

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

- A Swift runner implementing this contract on the Apple Core AI runtime
  (bundle loading, typed IO binding, tokenizer handling for LLMs).
- Standard seeded input corpora per modality (text prompts, audio clips,
  images) versioned in this repo so runs are reproducible.
- A decision on preprocessing parity for models with `processor_required`
  (the processor must be identical on both sides or the comparison is
  meaningless).
