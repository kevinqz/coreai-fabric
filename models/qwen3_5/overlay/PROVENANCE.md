# Provenance: qwen3_5 decode overlay

Vendored verbatim from the `coreai-model-zoo` repository's proven conversion
overlay (the same overlay already shipping behind Ornith-9B / MiniCPM-V-4.6).

- Source repo: `coreai-model-zoo` (`/Users/kevinsaltarelli/Dev/Github/coreai-model-zoo`)
- Source commit SHA: `53a5047e08e38e5e712426a42868cca10b78dd95`
- Source paths copied (relative to
  `conversion/overlay/files/python/src/coreai_models/models/` in the zoo repo):
  - `macos/qwen3_5.py`
  - `macos/qwen3_5_config.py`
  - `macos/qwen3_5_gdn_metal.py`
  - `macos/qwen3_5_metal_kernels.py`
  - `ios/qwen3_5.py`
  - `ios/qwen3_5_ios.py`
- Copy date: 2026-07-09

vendored verbatim; local edits tracked below

## Local edits

(none)

## Environment prerequisites

- `coreai_torch` — the overlay imports `GatedDeltaUpdate` from
  `coreai_torch.composite_ops`. This package is present in fabric's `.venv`
  (`coreai_torch==0.4.1`, site-packages), so the import-check in Task 1.1
  succeeded without needing to install anything. No pinned version
  requirement was found recorded elsewhere in the zoo repo (no
  `pyproject.toml`/`requirements*.txt`/`.lock` reference); `0.4.1` is simply
  the version already installed in this environment. If `coreai_torch` is
  ever missing in a fresh environment, install it before importing this
  overlay.
- Import also emits a benign runtime warning in this environment:
  `Skipping import of cpp extensions due to incompatible torch version.
  Please upgrade to torch >= 2.11.0 (found 2.9.0).` This does not affect the
  public API surface verified below and did not block any of the 6 expected
  symbols from resolving.

## Import verification (Task 1.1, Step 2)

```
.venv/bin/python -c "import sys; sys.path.insert(0,'models/qwen3_5/overlay/macos'); import qwen3_5 as q; print([n for n in ('Qwen3_5GatedDeltaNet','Qwen3_5FullAttention','Qwen3_5DecodeCore','build_decode_state','DECODE_STATE_NAMES','qwen3_5_config_from_hf') if hasattr(q,n)])"
```

Result: all 6 names resolved:
`['Qwen3_5GatedDeltaNet', 'Qwen3_5FullAttention', 'Qwen3_5DecodeCore', 'build_decode_state', 'DECODE_STATE_NAMES', 'qwen3_5_config_from_hf']`
