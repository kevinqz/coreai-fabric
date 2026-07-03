# Apple Core AI toolchain — verified notes (real hardware)

Everything in this document was measured or read from real installed
artifacts/sources on **2026-07-03**, on:

- Apple M4 Max, 64 GB RAM, **macOS 26.6** (Darwin 25.6.0)
- Xcode 26.5 — the ONLY installed SDK is `macosx26.5`
- Python 3.13.7 (venv), pip-installed stack below

Nothing here is guessed. Where something was not tested, it says so.

## Exact version pins (verified working together)

| Package | Version | Source | Role |
|---|---|---|---|
| `coreai-torch` | **0.4.1** | PyPI | PyTorch → Core AI IR converter **library** |
| `coreai-core` | **1.0.0b2** | PyPI (pulled by coreai-torch, `==` pin) | Compiler (MLIR) + **bundled runtime** |
| `torch` | 2.11.0 | PyPI (coreai-torch allows `>=2.8,<=2.11`) | export frontend |
| `transformers` | 4.57.3 | PyPI (the pin in coreai-torch's `[test]` extra and in every apple/coreai-models export script) | model sourcing |
| `coreai-models` | 0.1.0 | **GitHub only — NOT on PyPI** (checkout at commit `e203a0d`) | CLI exporters + Swift runners |
| `coreai-opt` | 0.2.1 | PyPI | compression — **used** for the 4bit production export (Finding 7) |

The static-graph driver run (Findings 3–5) used torch 2.11.0 / transformers
4.57.3. The production `coreai.llm.export` run (Finding 7) used a separate venv
with the coreai-models 0.1.0 checkout installed from source: torch 2.9.0,
transformers 4.57.6, coreai-opt 0.2.1 (same coreai-torch 0.4.1 / coreai-core
1.0.0b2).

## Finding 1 — there is NO `coreai-torch` executable

`coreai_torch-0.4.1-py3-none-any.whl` contains **no `entry_points.txt`** and
no console scripts. It is a library:

```python
from coreai_torch import TorchConverter, get_decomp_table
ep = torch.export.export(model, args=...).run_decompositions(get_decomp_table())
program = TorchConverter().add_exported_program(ep).to_coreai()
program.optimize()
program.save_asset(Path("out.aimodel"))   # AIProgram.save_asset — writes the bundle
```

The original `convert.py:build_command` assumption (`coreai-torch export <repo>
--output --precision --quantization --compute-units --revision`) was wrong on
every count: wrong executable, wrong subcommand, and none of those flags exist.

## Finding 2 — the real CLIs live in apple/coreai-models (not on PyPI)

`pip index versions coreai-models` → *no matching distribution*. The repo
(BSD-3, tag 0.1.0) defines console scripts in `python/pyproject.toml`:

```
coreai.llm.export       coreai.llm.eval       coreai.vlm.export
coreai.diffusion.export coreai.model.registry
```

`coreai.llm.export` verified argparse surface (read from
`python/src/coreai_models/llm/export.py`):

```
coreai.llm.export <short-name|owner/name>
    --platform {macOS,iOS}                 (default macOS)
    --compression <preset|none> | --compression-config <coreai-opt yaml>
    --compute-precision {float16,bfloat16,float32}
    --max-context-length N
    --output-dir DIR --output-name NAME
    --num-layers N --overwrite --experimental
    --list-presets --list-models --dry-run -v
```

- **No `--revision` flag** — upstream pinning is not supported by Apple's CLI.
- **No `--version` flag** — `tool --version` prints usage to stderr, exit 2.
- Output layout (`export/pipeline.py`): `<output-dir>/<output-name>/` contains
  `<output-name>.aimodel/` + `tokenizer/` + a bundle-level `metadata.json`
  (`metadata_version: "0.2"`, `kind: llm`, `assets.main`, `language.*`).
- Registry preset for the fabric seed: `qwen3-0.6b` → Qwen/Qwen3-0.6B,
  macOS, 4bit, float16, ctx 8192.
- Non-LLM families (whisper, depth-anything, clip, sam3, yolo, …) are
  converted by standalone **PEP 723 scripts** `models/<family>/export.py`
  (run via `uv run`), each pinning `coreai-torch==0.4.1`. Their flags differ
  per script; e.g. depth-anything supports **only**
  `--model depth-anything/da3-small --dtype float32`.

## Finding 3 — conversion works on macOS 26 (macOS 27 NOT required)

Measured end-to-end on this Mac (macOS 26.6):

1. Smoke model (`nn.Linear+ReLU`): export → convert → `save_asset` → OK.
2. **Qwen/Qwen3-0.6B** at pinned revision `c1899de2…`: full conversion via
   `coreai-fabric-llm-export` (static seq-len 96, float16) → **1.1 GB
   `.aimodel`** in ~3 minutes. See `docs/validation-log.md`.

The macOS-27 platform requirement in apple/coreai-models' `Package.swift`
applies to the **Swift runners** (CoreAILM etc.), not to conversion.

## Finding 4 — the PyPI runtime EXECUTES assets on macOS 26

Unexpected and load-bearing: `coreai-core`'s wheel bundles the Core AI
runtime. `coreai.runtime.AIModel.load(path)` (async) loads AND specializes a
saved `.aimodel` on macOS 26.6, and `InferenceFunction.__call__` executes it
(`await fn(inputs={"x": NDArray(np_array)})`). Smoke-model output matched
PyTorch with cosine 1.0. Consequence: **Gate B needs neither a Swift runner
nor macOS 27** — fabric now ships `coreai-fabric-parity-runner` on this API.

## Finding 5 — the real `.aimodel` inventory and metadata keys

A real asset directory (verified by producing one, consistent with all 354
`main.mlirb` files indexed in coreai-catalog's artifacts.yaml):

```
<name>.aimodel/
├── main.mlirb      # program bytecode (the big file)
├── main.hash       # sha256 of main.mlirb (32 raw bytes)
└── metadata.json   # {"creationDate": "...", "assetVersion": "2.0",
                    #  "producer": "coreai-core 1.0.0b2"}
```

- The format-version key is **`assetVersion`** (observed `"2.0"`), not
  `format_version` — Gate A now checks both spellings.
- Cross-checked against an externally published community bundle
  (bryanbblewis11/RealESRGAN-x4v3-CoreAI, digest-matched to the catalog):
  identical three-file inventory, but its `metadata.json` contains ONLY
  `assetVersion` — `creationDate`/`producer` are not guaranteed across
  publishers.
- `AIProgram.save_asset(path, metadata=None, minimum_os=OSVersion.v27)`:
  `OSVersion` has **exactly one member, `v27`** — every asset produced by
  coreai-core 1.0.0b2 declares minimum OS 27 for deployment.
- Runtime API surface (verified): `AIModelAsset.load/is_valid/metadata/summary`,
  `AIModel.load` → `load_function(name)` → descriptor
  (`desc.input_names`, `desc.input_descriptor(name)` → `.shape/.dtype`).
  Note: the converter narrows int64 token ids to **int32** graph inputs.

## Finding 6 — what fabric can and cannot drive (honest boundary)

| Path | Status |
|---|---|
| `coreai-fabric-llm-export` (this repo, `[convert]` extra) | **Validated on real hardware** — static logits graph, `--compression none` only |
| `coreai.llm.export` (apple/coreai-models checkout) | **Validated on real hardware** — produced a 320 MB 4bit KV-cache chat asset, Gate A passes (Finding 7) |
| `models/<family>/export.py` scripts | Interfaces verified from source; not executed here (DA3 needs a third-party git dependency) |
| Quantized/palettized exports (`coreai-opt`) | **Done** — the production export above is 4bit via the registry preset (Finding 7) |
| Swift runners (CoreAILM, …) | Cannot build or run on this Mac: `Package.swift` requires macOS 27; only SDK here is macosx26.5 |
| `coreai.llm.eval` (Gate B benchmark evaluator) | **Stub** in coreai-models 0.1.0 — prints "Evaluation support is coming soon" (Finding 7) |

## Finding 7 — the PRODUCTION `coreai.llm.export` asset (executed, measured)

Executed later the same day from a checkout venv (see `docs/validation-log.md`,
2026-07-03 21:09 UTC, for the verbatim run). Key INTERFACE facts discovered:

- **The production asset is quantized and STATEFUL.** Its runtime descriptor
  is fundamentally different from the static driver asset (Finding 5):

  ```
  input_names : ['input_ids', 'position_ids']   # driver: ['input_ids'] only
  output_names: ['logits']
  state_names : ['keyCache', 'valueCache']       # driver: none — this is a KV cache
  input_ids   : shape [1, -1] int32              # DYNAMIC seq len; driver: static [1, 96]
  logits      : shape [1, -1, 151936] float16
  ```

  Consequence: fabric's static-graph `parity-runner` **cannot drive it** (a
  plain forward raises "Missing state view for keyCache"). The runner detects
  `desc.state_names` and returns `not_run`.
- **Registry short-name resolves Apple's tested preset.** `coreai.llm.export
  qwen3-0.6b` (positional short-name, no `--compute-precision`/`--compression`)
  produced a **4bit / 8192-ctx** asset — verified against
  `coreai.model.registry --list-models`. A raw `owner/name` id instead needs
  `--experimental` + explicit `--compute-precision`.
- **The asset embeds the tokenizer + chat template.** `<name>.aimodel/` plus a
  sibling `tokenizer/` (7 files incl. `chat_template.jinja` — the real Qwen3
  tool-calling template) and richer `metadata.json` (6 keys: adds `license`,
  `author`, `description`). The on-device runner needs nothing else.
- **`coreai.llm.eval` is a STUB.** The KV-cache-aware benchmark evaluator that
  Gate B (`benchmark_accuracy`) would shell to prints "Evaluation support is
  coming soon" in 0.1.0. So production Gate B is blocked upstream, and fabric
  reports `not_run` rather than faking a number. This is the one gap between
  "production conversion works" and "production conversion verified".
