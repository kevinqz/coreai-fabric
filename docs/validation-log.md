# Validation log — real-hardware runs

Chronological record of what was actually executed. Companion to
`docs/toolchain-notes.md` (which records the discovered interfaces).

## 2026-07-03 — qwen3-0.6b converted and parity-verified on macOS 26.6

**Environment:** Apple M4 Max, 64 GB RAM, macOS 26.6 (Darwin 25.6.0),
Python 3.13.7. Stack: coreai-torch 0.4.1, coreai-core 1.0.0b2, torch 2.11.0,
transformers 4.57.3 (all PyPI). Upstream weights: Qwen/Qwen3-0.6B at pinned
revision `c1899de289a04d12100db370d81485cdf75e47ca` (1.4 GB download).

**Outcome: (a) — conversion succeeds on macOS 26.** The toolchain does NOT
require macOS 27 to convert, and the Python runtime executes the produced
asset on macOS 26 too. macOS 27 remains the deployment floor: coreai-core
1.0.0b2 can only serialize assets with `minimum_os v27`, and the
apple/coreai-models Swift runners declare macOS/iOS 27 in `Package.swift`
(this Mac's only SDK is macosx26.5, so the Swift side can't even build here).

### convert (real run, official CLI)

```
$ coreai-fabric convert qwen3-0.6b
running: coreai-fabric-llm-export Qwen/Qwen3-0.6B --output-dir .../build \
  --output-name qwen3-0.6b --compute-precision float16 --compression none \
  --overwrite --revision c1899de289a04d12100db370d81485cdf75e47ca --platform macOS
ok: conversion manifest written: build/qwen3-0.6b/conversion-manifest.json
ok: recipe status -> converted (qwen3-0.6b.yaml)
```

Conversion manifest (verbatim, paths abbreviated):

```json
{
  "recipe_id": "qwen3-0.6b",
  "tool": "coreai-fabric-llm-export",
  "tool_version": "coreai-fabric-llm-export 0.1.0",
  "converter_stack": {
    "coreai-torch": "0.4.1",
    "coreai-core": "1.0.0b2",
    "torch": "2.11.0",
    "transformers": "4.57.3"
  },
  "input": {
    "hf_repo": "Qwen/Qwen3-0.6B",
    "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "revision_pinned_by_tool": true
  },
  "started_at": "2026-07-03T13:17:13+00:00",
  "finished_at": "2026-07-03T13:17:25+00:00",
  "output_bundle": "build/qwen3-0.6b/qwen3-0.6b.aimodel",
  "asset_minimum_os": "27"
}
```

(The 12-second wall time is a warm re-run through the official CLI — model
weights and torch.export caches were hot from the first driver run minutes
earlier, which took ~3 minutes cold.)

### Real bundle inventory (measured)

```
build/qwen3-0.6b/
├── qwen3-0.6b.aimodel/
│   ├── main.mlirb      1,192,768,448 bytes   # program bytecode
│   ├── main.hash                  32 bytes   # sha256(main.mlirb)
│   └── metadata.json             105 bytes
├── tokenizer/           # upstream tokenizer (11 files)
└── metadata.json        # fabric bundle-level manifest
```

`qwen3-0.6b.aimodel/metadata.json` (verbatim):

```json
{
  "producer" : "coreai-core 1.0.0b2",
  "assetVersion" : "2.0",
  "creationDate" : "20260703T131724Z"
}
```

This is the ground truth behind Gate A's `assetVersion` mapping and the
recipes' `expected.bundle_files` (`metadata.json`, `main.mlirb`, `main.hash`).

### Cross-check against an externally published bundle (measured)

Downloaded the smallest catalog-indexed community artifact,
`bryanbblewis11/RealESRGAN-x4v3-CoreAI` @ `ae80a8b1…`
(`realesr-x4v3_float16_256.aimodel`, 2.5 MB; `main.mlirb` sha256 matches the
catalog's pinned digest `2ca0ec12…`):

- identical inventory: `main.mlirb`, `main.hash`, `metadata.json`;
- `metadata.json` is `{"assetVersion": "2.0"}` ONLY — `creationDate`/`producer`
  are not guaranteed across publishers, so Gate A must not require them
  (and does not);
- `AIModelAsset.is_valid` → True; summary exposes one `main` function.

Download deleted after inspection.

### Pre-conversion smoke checks (measured, same environment)

- Tiny `nn.Linear+ReLU` module: export → convert → `save_asset` →
  `AIModel.load` → inference; converted output vs PyTorch cosine **1.0**.
- `AIModelAsset.is_valid` → True on the produced asset; runtime function
  descriptor for the qwen3 bundle: `input_ids` int32 `[1, 96]` → `logits`
  (coreai-torch narrows int64 token ids to int32).

### verify (real run, official CLI, Gate A + Gate B)

```
$ COREAI_FABRIC_PARITY_RUNNER=coreai-fabric-parity-runner coreai-fabric verify qwen3-0.6b
  gate A [ok  ] bundle_exists: build/qwen3-0.6b/qwen3-0.6b.aimodel
  gate A [ok  ] bundle_files_present: 3 expected file(s) present
  gate A [ok  ] metadata_json_parses: metadata.json parses (3 top-level keys)
  gate A [ok  ] metadata_matches_recipe: 1 key(s) match
  gate B [failed] per_token_logit_cosine = 0.9966087344504636
report: build/qwen3-0.6b/parity-report.json (overall: failed)   # exit 1
```

**Gate A: PASSED** (all four checks, including the `assetVersion` ↔
`expected.format_version` match).

**Gate B: MEASURED, and honestly FAILED the 0.999 convention threshold.**
`parity-report.json` gate_b section (verbatim):

```json
{
  "metric": "per_token_logit_cosine",
  "threshold": 0.999,
  "tolerance": 0.0005,
  "value": 0.9966087344504636,
  "greedy_token_exact_required": true,
  "status": "failed",
  "greedy_token_exact": false,
  "n_inputs": 8,
  "decode_len": 64,
  "static_seq_len": 96,
  "reference_dtype": "float32",
  "runner": "coreai-fabric-parity-runner/0.1.0",
  "environment": {
    "os": "macOS 26.6",
    "chip": "Apple M4 Max",
    "machine": "arm64",
    "runtime_version": "1.0.0b2",
    "coreai_torch": "0.4.1",
    "torch": "2.11.0",
    "transformers": "4.57.3"
  }
}
```

Reading: the MINIMUM per-token cosine across 8 prompts × 64 greedy steps
(512 comparisons) between the float16 converted graph and the float32
upstream was 0.99661, and at least one greedy token diverged. The pipeline
behaved exactly as designed: verify exited 1 and the status did NOT advance
to verified. This is a real, useful data point — a plain float16 conversion
of Qwen3-0.6B does not clear the community's 0.999/greedy-exact bar against
an fp32 reference on this metric's worst case. Follow-ups worth measuring
(none of which were done here, so they are not claimed): bfloat16 or float32
conversion precision, comparing against an fp16 reference, and per-step
cosine distribution (the failure is the min, not the mean). Thresholds were
NOT tuned to make this pass.

### Honest boundaries of this validation

- The bundle is a STATIC (1, 96) logits graph — parity-verifiable and
  runnable, but not Apple's KV-cache chat-asset layout. Producing the
  production layout requires `coreai.llm.export` from an apple/coreai-models
  checkout, whose interface was verified from source but which was not
  executed in this session (sandbox policy blocked installing the checkout;
  the package is not on PyPI).
- whisper-large-v3-turbo and da3-small were NOT converted: their converters
  are per-model PEP 723 scripts in apple/coreai-models (da3 additionally
  pulls `depth-anything-3` from a third-party git repo). Their interfaces
  (flags, dtypes) were verified from the repo source and are recorded in the
  recipes.
- Gate B ran on the runtime bundled in the coreai-core wheel, not the
  on-device OS runtime (which needs macOS 27).
- The recipe status was reset to `draft` before committing: converted/verified
  describe disposable local `build/` state (this log is the durable record).
  The 1.1 GB bundle was deleted after verification — fabric never hosts
  weights.
