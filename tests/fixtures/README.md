# Vendored catalog schema snapshots

`model.schema.json`, `artifact.schema.json`, and `source.schema.json` are
verbatim SNAPSHOTS of the coreai-catalog schemas, refreshed 2026-07-03 after
the catalog's P1 wave (they now include `bundle_kind`, `min_os`, `io_contract`,
and `upstream_repo` on the model schema).

They exist ONLY so the register generator can be tested OFFLINE. They are NOT
the drift guard — vendored snapshots always re-stale over time. The real drift
guard is **`scripts/cross_contract_check.py`** (wired into
`.github/workflows/cross-contract.yml`): it clones the LIVE catalog and runs
fabric's real register output through the catalog's own validate + audit +
io_contract invariant tests on every push, PR, and weekly. That job — not these
snapshots — is what catches a catalog schema/invariant change fabric no longer
satisfies. At register time the real schemas are read from `--catalog-path`,
never from here.
