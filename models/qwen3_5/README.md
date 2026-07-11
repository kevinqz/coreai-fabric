# models/qwen3_5

Bespoke Qwen3.5 VLM exporter. Fabric does not run `.py` tools
(convert.py:194) — run `export.py` manually, drop the bundle at
`build/<id>/<id>.aimodel`, then `coreai-fabric verify <id>`.
Every optimization attempt is recorded via `ledger.py` (provenance/*.jsonl).
