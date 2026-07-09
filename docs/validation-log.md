# Validation log — real-hardware runs

Chronological record of what was actually executed. Companion to
`docs/toolchain-notes.md` (which records the discovered interfaces).

## 2026-07-03 — qwen3-0.6b converted and parity-verified on macOS 26

**Environment:** Apple Silicon, macOS 26,
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
    "os": "macOS 26",
    "chip": "Apple Silicon",
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
  checkout (not on PyPI). That production run was executed LATER THE SAME DAY
  — see the next section, "PRODUCTION export via apple/coreai-models" — which
  supersedes the earlier note here that it had not been run.
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

## 2026-07-03 (21:09 UTC) — PRODUCTION export via apple/coreai-models `coreai.llm.export`

The production path the ecosystem actually deploys. Unlike the static-graph
driver run above, this produces Apple's **stateful KV-cache chat asset** —
the layout the on-device runner expects.

**Environment:** same Mac (Apple Silicon, macOS 26, arm64). A fresh
venv with the apple/coreai-models checkout installed from source
(`git clone` + `pip install ./coreai-models/python`; it is NOT on PyPI).
Checkout at commit `e203a0d`; installed stack: coreai-models 0.1.0,
coreai-torch 0.4.1, coreai-core 1.0.0b2, coreai-opt 0.2.1, torch 2.9.0,
transformers 4.57.6. macOS 27 was NOT required to build the asset (only to
run it on device).

### Registry preset (verbatim, `coreai.model.registry --list-models --type llm`)

```
qwen3-0.6b   macOS   4bit                            8192   Qwen/Qwen3-0.6B
qwen3-0.6b   iOS     qwen3_0_6b_mixed_4bit_8bit.yaml 4096   Qwen/Qwen3-0.6B
```

The registry SHORT-NAME (`qwen3-0.6b`) auto-resolves Apple's TESTED preset —
4bit / 8192-token context on macOS. This is why the production recipe passes
`apple_registry_name: qwen3-0.6b` (positional) and does NOT pass
`--compute-precision`/`--compression`: the preset must win. A raw HF id would
instead require `--experimental` + explicit precision.

### convert (real run, Apple's production CLI)

```
coreai.llm.export qwen3-0.6b --output-dir <build> --output-name qwen3-0.6b-prod --overwrite
```

(The probe used `--output-name qwen3-0.6b-prod` to avoid clobbering the
static-graph build above; the seed recipe uses `--output-name qwen3-0.6b`.)

### Real production bundle inventory (measured)

```
build/qwen3-0.6b-prod/
├── qwen3-0.6b-prod.aimodel/
│   ├── main.mlirb      335,979,906 bytes   # 4bit-quantized program (320 MB)
│   ├── main.hash                32 bytes
│   └── metadata.json           307 bytes   # 6 top-level keys (see below)
└── tokenizer/                              # EMBEDDED, 7 files:
    ├── chat_template.jinja   4,168 bytes   # real Qwen3 template (tool-calling)
    ├── tokenizer.json   11,422,654 bytes
    ├── vocab.json        2,776,833 bytes
    ├── merges.txt        1,671,853 bytes
    ├── tokenizer_config.json 5,404 bytes
    ├── special_tokens_map.json 613 bytes
    └── added_tokens.json       707 bytes
```

Two facts the static-graph driver's asset did NOT have: the 4bit asset is
**320 MB** vs the driver's 1.19 GB float16 graph, and it carries an
**embedded tokenizer + chat template** so the on-device runner needs nothing
else.

`qwen3-0.6b-prod.aimodel/metadata.json` (verbatim):

```json
{
  "assetVersion" : "2.0",
  "license" : "Apache-2.0",
  "producer" : "coreai-core 1.0.0b2",
  "creationDate" : "20260703T210920Z",
  "author" : "Qwen Team",
  "description" : "Qwen3-0.6B is a 0.6B-parameter causal language model from the Qwen3 family. Source: https://huggingface.co/Qwen/Qwen3-0.6B"
}
```

### Runtime function descriptor (verbatim — this is a STATEFUL asset)

Loaded via `AIModel.load(...).load_function("main").desc` on the coreai-core
runtime:

```
input_names : ['input_ids', 'position_ids']
output_names: ['logits']
state_names : ['keyCache', 'valueCache']        # <- KV cache: STATEFUL
  in  input_ids   : shape=[1, -1]         dtype=int32
  in  position_ids: shape=[1, -1]         dtype=int32
  out logits      : shape=[1, -1, 151936] dtype=float16
```

This is the crux for Gate B: the asset carries a KV cache (`state_names`) and
takes a DYNAMIC sequence length (`[1, -1]`). Fabric's `parity-runner` is a
static-graph runner (fixed `(1, seq_len)` `input_ids` → full-sequence
`logits`, `use_cache=False`); it literally cannot drive this — a plain forward
raises "Missing state view for keyCache". The runner now detects
`desc.state_names` and returns `not_run` with that reason instead of crashing.

### Gate A (real run vs the production asset) — PASSED

```
GATE A: passed
  [passed] bundle_exists: build/qwen3-0.6b-prod/qwen3-0.6b-prod.aimodel
  [passed] bundle_files_present: 3 expected file(s) present
  [passed] metadata_json_parses: metadata.json parses (6 top-level keys)
  [passed] metadata_matches_recipe: 1 key(s) match     # assetVersion 2.0 == expected 2.0
```

### Gate B — not_run, blocked UPSTREAM (honest)

```
GATE B: not_run
reason: Gate B for a production coreai.llm.export asset is benchmark accuracy
        vs upstream (e.g. tinyMMLU) — the correct metric for a quantized
        asset, whose raw logits legitimately diverge. The only conforming
        evaluator is Apple's coreai.llm.eval, which ships as a stub in
        coreai-models 0.1.0 ("Evaluation support is coming soon"), and the
        stateful KV-cache asset cannot be scored by a static-graph runner.
```

Two independent reasons Gate B can't produce a number today, both real:

1. **Wrong metric for a quantized asset.** Raw per-token logit cosine vs an
   fp32 reference is meaningful for a float16 graph (the driver run above), but
   a 4bit asset legitimately diverges in logits while preserving task
   accuracy. The correct Gate B is a benchmark-accuracy eval (e.g. tinyMMLU
   within tolerance of upstream).
2. **The upstream evaluator is a stub.** Apple's `coreai.llm.eval` — the
   KV-cache-aware benchmark evaluator — prints "Evaluation support is coming
   soon" in coreai-models 0.1.0. There is no conforming evaluator to shell to.

So the honest state of the production path: **conversion works and is wired
end-to-end** (recipe → `coreai.llm.export` → real 4bit KV-cache chat asset →
Gate A passes), and **Gate B stays `not_run` until Apple ships their
evaluator**. Fabric never fakes a parity number and never records this as a
failure — a metric that can't run yet is not a parity failure.

The 320 MB bundle + the ~2 GB scratch venv/checkout were deleted after these
measurements — fabric never hosts weights.

---

## 2026-07-08 — LingBot-World V2 14B causal-fast: VAE decoder (FIRST video lane)

First generative-video conversion in the fleet. Deployable core = the
`AutoencoderKLWan` video VAE decoder (pure conv, no attention), exported as one
static-size fp32 `.aimodel` via `models/lingbotworld/export.py`. Ran end-to-end
in `.venv` (coreai-torch 0.4.1, coremltools 9.0, diffusers 0.37.1):

- Weights: `vae/diffusion_pytorch_model.safetensors` @ 59cccf49 (~508MB fp32),
  194 tensors load clean into `AutoencoderKLWan(**cfg)` — 0 missing / 0 unexpected.
- Graph: first-chunk decode (`num_frame=1`, `use_tiling` off). Latent
  `[1,16,1,60,104]` → frames `[1,3,1,480,832]` (true causal-fast 480×832).
- `torch.export` → `run_decompositions(get_decomp_table())` →
  `TorchConverter.add_exported_program` → `to_coreai().optimize()` →
  `save_asset`. Bundle ~286MB: `metadata.json` + `main.mlirb` + `main.hash`.
  Single program — no graph-split (well under the 0x10004 ceiling).
- Runtime: `coreai.runtime.AIModel.load` → `load_function("vae_decode")` → LOADS OK.
- **Gate B (graph_output_cosine, fp32 ref vs fp32 asset, n_obs=8): min
  0.99999999999941 / median 0.99999999999941 / mean 0.99999999999941.** Crushes
  the 0.999 threshold.

Status `verified`, index-only (CC-BY-NC-SA → no republish). Asset + parity JSON
under `build/` (gitignored); weights not hosted. Follow-ups: (1) expose the WAN
causal `feat_cache` as graph I/O for streaming multi-chunk decode (prefix-K/V
technique); (2) DiT 14B graph-split (research-grade, host-owned).

---

## 2026-07-08 — Backlog sweep: pulpie-orange-small + LingBot-Video family

**pulpie-orange-small** (EuroBERT-210m token classifier, `feyninc/pulpie-orange-small`
@ 15ead335, **cc-by-nc-4.0** → index-only, unlike the apache base/large). Built via
`models/eurobert/export.py` (one static-seq-64 `.aimodel`, `(input_ids,
attention_mask) -> [1,64,2]`) + `models/eurobert/parity.py`. **Gate B
(graph_output_cosine, n=8): min 0.99999999999174 / median 0.99999999999451.** LOADS OK.

**LingBot-Video family VAE decoder** (`robbyant/lingbot-video-dense-1.3b` @ f9789a7d,
**apache-2.0 → publishable**). VAE is `AutoencoderKLWan`, and its `vae/` safetensors
is **byte-identical (SHA256 d6e524b3…) to lingbot-world-v2's VAE** — the whole LingBot
video family (world-v2 14B, video-dense-1.3b, video-moe-30b-a3b) ships ONE shared VAE.
Built via `models/lingbotvideo/export.py` (latent [1,16,1,60,104] → frames
[1,3,1,480,832], one fp32 program ~286MB). **Gate B (n=8): min/median
0.99999999999941.** LOADS OK. Same number as the world-v2 build, as expected from the
identical weights. This asset is the deployable VAE decoder for all three members;
Apache-2.0 makes it publishable (kevinqz/LingBot-Video-Dense-1.3B-CoreAI).

Assets + parity JSON under build/ (gitignored). Recipes: pulpie-orange-small (verified,
index-only), lingbot-video-dense-1.3b (verified, publishable), lingbot-video-moe-30b-a3b
(draft — VAE verified by identity, 30B MoE DiT = research-grade). The
lingbot-video-rewriter-lora is a prompt-rewriter LoRA, out of scope for a .aimodel lane.
