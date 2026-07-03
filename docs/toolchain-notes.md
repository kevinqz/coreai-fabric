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
| `coreai-models` | 0.1.0 | **GitHub only — NOT on PyPI** | CLI exporters + Swift runners |
| `coreai-opt` | 0.2.1 | PyPI (not installed here; required only for quantized exports) | compression |

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
| `coreai.llm.export` (apple/coreai-models checkout) | Interface verified from source; **not executed here** (checkout installation was outside this session's sandbox policy) |
| `models/<family>/export.py` scripts | Interfaces verified from source; not executed here (same reason + DA3 needs a third-party git dependency) |
| Quantized/palettized exports (`coreai-opt`) | Not attempted |
| Swift runners (CoreAILM, …) | Cannot build or run on this Mac: `Package.swift` requires macOS 27; only SDK here is macosx26.5 |
