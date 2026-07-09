# attempts/ — committed conversion failure substrate (RFC Phase 1, F8)

Every `coreai-fabric run <id>` appends one JSONL record here — including
**failures**. This is the weakness-mining loop's data substrate: failed exports
previously left zero structured trace, and fabric never saw the 36/53 script-tool
runs.

Each record:
```json
{"ts": "...", "recipe": "<id>", "stage": "convert", "tool": "...", "exit": 1,
 "toolchain": {"coreai_torch": "0.4.1", ...},
 "error_signature": "0x10004", "error_tail": "...",
 "envelope": {"precision": "fp16", "quantization": "none", "tool": "..."},
 "outcome": "blocked"}
```

`error_signature` is a distilled class (see coreai_fabric/error_signatures.py):
`0x10004`, `complex128`, `OOM`, `parity_below_threshold`, … `unclassified` means
the table needs a new entry — a `/reflect` loop input.

This directory is COMMITTED (unlike build/). Records are append-only.
