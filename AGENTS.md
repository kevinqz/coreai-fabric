# Agent Instructions for coreai-fabric

This is the operating manual. Every step below is executable by an agent with
a shell; nothing requires out-of-band human knowledge. Where a step has a
hard prerequisite (macOS + Apple toolchain, HF token, gh auth) it is stated
explicitly, with the exact failure you will see without it.

## What this repo is

coreai-fabric is the conversion pipeline that turns upstream Hugging Face
models into provenance-verified Apple Core AI `.aimodel` artifacts, published
to the publisher's OWN Hugging Face namespace, then indexed by
[coreai-catalog](https://github.com/kevinqz/coreai-catalog).

- **Source of truth:** `recipes/*.yaml`, one recipe per artifact, validated
  against `schema/recipe.schema.json`.
- **Fabric never hosts weights.** `build/` is gitignored; published bytes live
  in each publisher's HF repo. The catalog stays index-not-host.
- **Never fabricate facts.** If a field is unknowable before conversion
  (runner, device support, format_version), it is absent or `unknown`.

## The pipeline

```
new -> validate -> convert -> verify -> publish -> register
draft            converted   verified  published  registered
```

Each command advances `status:` in the recipe. Statuses are only advanced by
the commands themselves — never edit `status` or `published:` by hand.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
coreai-fabric validate
```

For publishing you also need the HF extra and a logged-in token:

```bash
pip install -e ".[hf,test]"
hf auth login
```

## 1. new — scaffold a recipe

```bash
coreai-fabric new Qwen/Qwen3-0.6B
```

Fetches license, `pipeline_tag`, revision sha, and size from the HF API and
writes `recipes/<id>.yaml` with `status: draft`. Override anything via flags:

```bash
coreai-fabric new openai/whisper-large-v3-turbo --id whisper-large-v3-turbo --namespace coreai-community --precision fp16
```

Offline (fabric never invents a license, so you must supply one):

```bash
coreai-fabric new Qwen/Qwen3-0.6B --offline --license apache-2.0 --license-terms permissive
```

- If the `pipeline_tag` has no honest capability mapping, the `catalog:` block
  is omitted and you must add it by hand before `register`.
- Licenses outside the permissive allowlist are triaged `review_required`
  automatically.

## 2. validate — schema + license triage

```bash
coreai-fabric validate
coreai-fabric validate qwen3-0.6b
```

Errors are AGGREGATED (all findings in one pass, each with a fix hint), never
fail-fast. Checks: JSON-schema conformance, id/filename match,
status/published consistency, and license triage (`review_required` upstreams
are flagged; claiming `permissive` for a non-allowlisted license is an error).
Exit code 0 = no errors (warnings allowed), 1 = errors.

## 3. convert — run the Apple toolchain

**Prerequisite: macOS on Apple Silicon with Apple's converter installed**
(see the apple/coreai-torch and apple/coreai-models repos). Without it,
convert fails honestly with install instructions — there is no simulation
mode. CI never converts.

```bash
coreai-fabric convert qwen3-0.6b
coreai-fabric convert qwen3-0.6b --print-command
```

`--print-command` shows the exact converter invocation without running it.
NOTE: the converter flag layout in `coreai_fabric/convert.py:build_command`
is TODO-marked — it has not been verified against a real toolchain install.
If your toolchain differs, fix `build_command` (one function) and record the
working version in the recipe's `conversion.min_tool_version`.

Outputs on success:
- `build/<id>/<id>.aimodel/` — the bundle
- `build/<id>/conversion-manifest.json` — tool, version, exact command,
  timestamps, pinned upstream revision
- recipe `status: converted`

## 4. verify — Gate A + Gate B

```bash
coreai-fabric verify qwen3-0.6b
```

- **Gate A (runs anywhere):** bundle directory exists, `expected.bundle_files`
  present, `metadata.json` parses, overlapping metadata keys match the recipe.
- **Gate B (requires macOS + Apple Core AI runtime):** numeric parity vs the
  upstream — cosine metric/threshold from the recipe (`parity.gate_b`).
  Protocol: `docs/parity-protocol.md`. Fabric does not yet ship the Swift
  runner; without one configured (`COREAI_FABRIC_PARITY_RUNNER`), Gate B is
  recorded `not_run` — never faked.

Writes `build/<id>/parity-report.json`. Status advances to `verified` only
when BOTH gates pass (exit 0). Gate A pass + Gate B not_run = exit 2
("partial") and the status does not advance.

## 5. publish — upload to the publisher's own HF namespace

**Prerequisites:** `pip install -e ".[hf]"`, `hf auth login`, and write access
to the target namespace in the recipe's `publish:` block.

```bash
coreai-fabric publish qwen3-0.6b --dry-run
coreai-fabric publish qwen3-0.6b
```

Refuses to publish if: no bundle, no parity report, Gate A failed, Gate B not
passed (override consciously with `--allow-unverified-parity` — the model card
records it), or license is `review_required` without
`--acknowledge-license-review`.

Uploads the bundle + generated model card (provenance, base_model,
converted_by, gate outputs) + `parity-report.json` + `conversion-manifest.json`,
then writes the `published: {hf_repo, revision, date}` block into the recipe.

## 6. register — index it in coreai-catalog

**Prerequisites:** a local clone of coreai-catalog whose schemas include the
fabric field contract, plus `gh auth login` for the PR step.

Always dry-run first:

```bash
coreai-fabric register qwen3-0.6b --catalog-path ../coreai-catalog --dry-run
```

Then for real:

```bash
coreai-fabric register qwen3-0.6b --catalog-path ../coreai-catalog
```

What it does:
1. Fetches per-file sha256 digests of the published repo at the pinned
   revision from the HF API (LFS oids are sha256; non-LFS files are downloaded
   and hashed — no fabricated digests).
2. Generates the `catalog.yaml` model entry (`source_group: fabric`,
   `source_path` = the recipe URL) and the `artifacts.yaml` artifact entry
   (huggingface-only — no github block; `huggingface.revision` +
   `huggingface.files[]`; `provenance.converted_by` + `recipe_source: fabric`),
   plus the `sources.yaml` record for `coreai-fabric`.
3. Validates both entries against the catalog clone's OWN schemas — aggregated
   errors, including a hint if the clone predates the fabric contract.
4. Applies the entries to the clone (append + `metadata.count` bump), runs the
   catalog's `scripts/validate.py`, `generate.py`, `audit.py` locally so the
   PR arrives green, then pushes branch `fabric/add-<id>` and opens a PR via
   `gh pr create`.

## Inventory

```bash
coreai-fabric list
coreai-fabric status
coreai-fabric status qwen3-0.6b
```

## Failure modes (what you will actually see)

| Failure | Cause | Fix |
|---|---|---|
| `converter 'coreai-torch' not found on PATH` | No Apple toolchain (e.g. Linux/CI) | Install on macOS per apple/coreai-torch; or set `COREAI_FABRIC_TOOL` |
| `gate B ... not_run` | No parity runner | Provide a runner implementing `docs/parity-protocol.md`; set `COREAI_FABRIC_PARITY_RUNNER` |
| `license ... is not on the fabric permissive allowlist` | `license_terms: permissive` overclaim | Set `review_required` |
| `upstream license is review_required` at publish | Triage flag | Human review, then `--acknowledge-license-review` |
| `huggingface_hub is not installed` | Missing extra | `pip install "coreai-fabric[hf]"` |
| `generated entries do not validate against .../schema` | Catalog clone predates the fabric contract | Pull/update the catalog clone |
| catalog `audit.py` flags `source_group=fabric vs artifact group=external` | Catalog audit not yet taught the fabric pairing (register aborts before the PR — by design) | Update the catalog clone once its audit learns the mapping |
| `catalog clone ... has uncommitted changes` | Dirty working tree | Commit or stash in the catalog clone |
| `gh pr create failed` | gh not authenticated | `gh auth login` |

## Repo conventions

- Recipe filename must equal the recipe `id` (`recipes/<id>.yaml`).
- Recipe ids must not collide with existing coreai-catalog model ids (the
  recipe id becomes the catalog model id).
- `build/` is disposable and gitignored. Reports intended to persist are
  published to the artifact's HF repo, not committed here.
- Tests: `python -m pytest`. CI (`.github/workflows/validate.yml`) runs
  validate + pytest + a doc-check that every `coreai-fabric` command in this
  file parses. It never converts (no Apple toolchain on runners).
