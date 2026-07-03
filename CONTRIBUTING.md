# Contributing to coreai-fabric

Contributions are **recipes** and **parity reports**. This repo is agent-first:
every acceptance rule below is checkable by running a command, and PRs are
expected to arrive green.

## What a contribution looks like

### A new recipe (most common)

1. Scaffold it:
   ```bash
   coreai-fabric new the-org/the-model
   ```
2. Review every generated field. In particular:
   - `upstream.license` / `license_terms` — must reflect the upstream repo's
     actual declaration. If the license is not on the permissive allowlist it
     must be `review_required`.
   - `catalog:` block — required before the recipe can ever be registered;
     use honest values and `unknown` where you do not know.
   - `conversion:` parameters — what you intend to run, not what you wish.
3. Validate:
   ```bash
   coreai-fabric validate
   ```
4. Open a PR containing ONLY `recipes/<id>.yaml`. Recipes land as
   `status: draft` — you do not need macOS or the Apple toolchain to
   contribute a recipe.

### A conversion + parity report (advances a draft recipe)

If you have macOS + the Apple toolchain, run the loop:

```bash
coreai-fabric convert the-id
coreai-fabric verify the-id
coreai-fabric publish the-id
```

Publishing targets YOUR Hugging Face namespace (edit
`publish.hf_target_namespace` in the recipe — fabric never hosts weights and
never asks you to upload to someone else's account). Your PR then updates the
recipe (`status`, `published:` block, any corrected `conversion` /
`min_tool_version` facts) — the parity and conversion reports live in your
published HF repo, not in this git repo.

### Fixing the toolchain adapter

`coreai_fabric/convert.py:build_command` maps the recipe contract onto an
assumed converter interface and is explicitly TODO-marked. If you have the
real toolchain and the flags differ, fixing that one function (with the
observed `--version` output quoted in the PR) is one of the most valuable
contributions possible.

## Rules (all checkable)

| Rule | Check |
|---|---|
| Recipe validates | `coreai-fabric validate` exits 0 |
| License triage clean | `validate` reports no license errors; `review_required` is honest, not hidden |
| Upstream resolves | `upstream.hf_repo` exists on HF (CI checks; or `coreai-fabric new <repo> --force` re-resolves) |
| Filename = id | `recipes/<id>.yaml` |
| No weights in git | nothing under `build/` committed; no binary blobs |
| No fabricated facts | unknowable fields absent or `unknown`; `status`/`published` only advanced by the commands |
| Tests pass | `python -m pytest` |

See [GOVERNANCE.md](GOVERNANCE.md) for exactly what "mergeable" means.

## Dev setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
python -m pytest
```
