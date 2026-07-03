# Vendored catalog schema snapshots

`model.schema.json`, `artifact.schema.json`, and `source.schema.json` are
verbatim SNAPSHOTS of the coreai-catalog schemas, taken 2026-07-03 after the
shared fabric field contract landed there:

- `model.schema.json`: `source_group` enum includes `fabric`.
- `artifact.schema.json`: `github` is optional via `anyOf(github,
  huggingface)`; `huggingface` has optional `revision` (40-hex sha) and
  `files[] {path, sha256, size_bytes}`; top-level has optional `provenance`
  (`converted_by {tool, version, recipe_url}`, `recipe_source` enum
  `[apple-official, zoo-port, fabric, independent]`, `format_version`) and
  `mirrors[]`.

They exist ONLY so the register generator can be tested offline. The living
source of truth is `schema/` in https://github.com/kevinqz/coreai-catalog —
at register time the real schemas are read from `--catalog-path`, never from
here. If the catalog schemas change, refresh these snapshots.
