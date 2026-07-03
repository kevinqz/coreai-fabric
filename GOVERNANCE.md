# Governance

Small repo, explicit rules. A maintainer merges; an agent (or anyone) can
verify every condition below before asking for the merge. If all conditions
hold, the expectation is a merge — "mergeable" is a checklist, not a mood.

## Mergeable = ALL of:

1. **Recipe validates.**
   `coreai-fabric validate` exits 0 (schema conformance, id/filename match,
   status consistency, aggregated across all recipes).

2. **License triage clean.**
   No license errors from `validate`. `license_terms: permissive` only for
   allowlisted licenses (`coreai_fabric/recipes.py:PERMISSIVE_LICENSES`);
   everything else is `review_required` and stays flagged until a human
   reviews it. Triage labels are not legal advice.

3. **Upstream resolves.**
   `upstream.hf_repo` (at `upstream.revision` if pinned) exists on the
   Hugging Face Hub at review time.

4. **No digest conflicts.**
   For recipes with a `published:` block: the published repo at the pinned
   revision must exist, and its file digests must not contradict digests
   already recorded anywhere in this repo or in coreai-catalog for the same
   `hf_repo@revision`. Two recipes may not publish to the same target repo.

5. **CI green.**
   `.github/workflows/validate.yml` (validate + pytest + AGENTS.md doc-check)
   passes.

6. **No hosted weights.**
   The diff contains no model binaries. Fabric is a pipeline, never a host.

## Status advancement

`status:` and `published:` are written by the pipeline commands only. A PR
that hand-advances a recipe to `verified`/`published` without the
corresponding artifacts (published HF repo with parity + conversion reports)
is not mergeable.

## Roles

- **Maintainer:** Kevin Saltarelli (@kevinqz) — merge authority, license
  review sign-off.
- **Contributors:** anyone, human or agent. All acceptance criteria are
  runnable locally, so review requests should arrive pre-verified.

## Changes to the contract

`schema/recipe.schema.json` and the catalog field contract (what `register`
emits) are versioned interfaces: changes require a PR that also updates
AGENTS.md, the seed recipes, and the vendored test fixtures — CI enforces
that they stay in sync with themselves, and the coreai-catalog CI enforces
the other side.
