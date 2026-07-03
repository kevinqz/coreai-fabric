"""`coreai-fabric register` — generate coreai-catalog entries for a published
artifact and open a PR against github.com/kevinqz/coreai-catalog.

Implements the shared field contract between fabric and the catalog:
  - model entry: source_group: fabric
  - artifact entry: NO github block (huggingface-only provenance, allowed by
    the catalog's anyOf(github, huggingface)), huggingface.revision +
    huggingface.files digests, and a provenance block with
    converted_by {tool, version, recipe_url} + recipe_source: fabric.

Entries are validated against the catalog schemas read from --catalog-path,
so register fails loudly (not silently) if the target catalog clone predates
the contract.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from . import FABRIC_REPO_URL
from .convert import manifest_path
from .recipes import Recipe, commercial_use_for, find_recipe
from .util import dump_yaml, err, find_root, ok, today, warn, write_yaml
from .verify import parity_report_path

SOURCE_RECORD_ID = "coreai-fabric"


def _bool_or_unknown(value) -> object:
    return value if value in (True, False) else "unknown"


def _humanize_bytes(total: int) -> str:
    if total >= 1 << 30:
        return f"{total / (1 << 30):.1f}GB"
    if total >= 1 << 20:
        return f"{total / (1 << 20):.1f}MB"
    return f"{total}B"


def recipe_url(recipe: Recipe) -> str:
    return f"{FABRIC_REPO_URL}/blob/main/recipes/{recipe.id}.yaml"


def build_model_entry(recipe: Recipe, files: list[dict], *, notes_suffix: str = "") -> dict:
    catalog_block = recipe.data.get("catalog")
    if not catalog_block:
        raise SystemExit(
            f"error: recipe {recipe.id} has no catalog: block — register needs "
            "name/family/capabilities/modalities to generate the model entry. "
            "Add it to the recipe (see schema/recipe.schema.json)."
        )
    conv = recipe.data["conversion"]
    total_bytes = sum(f.get("size_bytes", 0) for f in files)
    # Runtime facts are declared by the recipe (fabric is the converter, so it is the
    # authoritative source). The catalog audit rejects `unknown` for these four; the
    # recipe schema requires them as real booleans, so they are always present here.
    rf = catalog_block["runtime_facts"]
    # min_os: the recipe may declare it; else default to the apple-core-ai floor
    # (macOS/iOS 27.0, grounded in apple/coreai-models Package.swift).
    min_os = catalog_block.get("min_os", {"macos": "27.0", "ios": "27.0"})
    entry: dict = {
        "id": recipe.id,
        "name": catalog_block["name"],
        "family": catalog_block["family"],
        "source_group": "fabric",
        "source_path": recipe_url(recipe),
        "artifact_ref": recipe.id,
        "capabilities": list(catalog_block["capabilities"]),
        "bundle_kind": catalog_block["bundle_kind"],
        "min_os": {"macos": min_os["macos"], "ios": min_os["ios"]},
        # Upstream repo (org/name) is the discovery dedup key; fabric holds it verbatim.
        "upstream_repo": recipe.data["upstream"]["hf_repo"],
        "modalities": {
            "input": list(catalog_block["modalities"]["input"]),
            "output": list(catalog_block["modalities"]["output"]),
        },
        "artifact": {"format": "aimodel", "availability": "available"},
        "size": {
            "parameters": catalog_block.get("parameters", "not_published"),
            "precision": conv["precision"],
            "quantization": conv["quantization"],
            "artifact_size": _humanize_bytes(total_bytes) if total_bytes else "not_published",
        },
        "runtime": {
            "runtime_name": "apple-core-ai",
            "runner": catalog_block.get("runner", "CoreAIRunner"),
            "stock_runtime": rf["stock_runtime"],
            "custom_kernel": rf["custom_kernel"],
            "patch_required": rf["patch_required"],
            "tokenizer_required": _bool_or_unknown(catalog_block.get("tokenizer_required")),
            "processor_required": _bool_or_unknown(catalog_block.get("processor_required")),
            "aot_required": rf["aot_required"],
        },
        # Device support is unknowable until someone runs the artifact on-device;
        # fabric never asserts it.
        "device_support": {
            "iphone": "unknown",
            "ipad": "unknown",
            "mac": "unknown",
            "mac_only": "unknown",
        },
        "license": {
            "name": recipe.data["upstream"]["license"],
            "commercial_use": commercial_use_for(recipe),
        },
        "status": "needs_review",
        "maturity": "experimental",
        "confidence": "medium",
        "sources": [SOURCE_RECORD_ID],
        "last_verified": today(),
        "notes": (
            f"Converted via coreai-fabric recipe {recipe.id} "
            f"({recipe_url(recipe)}).{notes_suffix}"
        ),
    }
    if catalog_block.get("architecture"):
        entry["architecture"] = catalog_block["architecture"]
    if isinstance(catalog_block.get("streaming"), bool):
        entry["streaming"] = catalog_block["streaming"]
    return entry


def build_artifact_entry(recipe: Recipe, files: list[dict], tool_version: str | None) -> dict:
    published = recipe.data.get("published")
    if not published:
        raise SystemExit(
            f"error: recipe {recipe.id} has no published block — run "
            f"`coreai-fabric publish {recipe.id}` first (register indexes "
            "published artifacts only)."
        )
    hf_repo = published["hf_repo"]
    owner, repo = hf_repo.split("/", 1)
    conv = recipe.data["conversion"]
    provenance: dict = {
        "converted_by": {
            "tool": conv["tool"],
            "version": tool_version or "unknown",
            "recipe_url": recipe_url(recipe),
        },
        "recipe_source": "fabric",
    }
    fmt = (recipe.data.get("expected") or {}).get("format_version")
    if fmt:
        provenance["format_version"] = fmt
    return {
        "id": recipe.id,
        # Per the shared field contract the artifact group enum is unchanged
        # (zoo/official/external/unknown) while the MODEL gains source_group
        # fabric; external is the honest artifact group for an independent
        # conversion. Known caveat: catalog audit category 2 compares
        # artifact.group with model.source_group, so the catalog-side audit
        # must learn the fabric<->external pairing before a register PR can
        # pass its local audit run (register aborts safely until then).
        "group": "external",
        # No github block: fabric provenance is huggingface-native. The catalog
        # artifact schema allows this via anyOf(github, huggingface) — no more
        # fabricated GitHub coordinates for HF-only conversions.
        "huggingface": {
            "owner": owner,
            "repo": repo,
            "url": f"https://huggingface.co/{hf_repo}",
            "revision": published["revision"],
            "files": files,
        },
        "provenance": provenance,
        "officiality": {
            "apple_export_recipe": False,
            "apple_hosted_artifact": False,
            "community_packaged": True,
        },
    }


def build_source_record() -> dict:
    """sources.yaml record the catalog needs so `sources: [coreai-fabric]`
    cross-references resolve. Shape follows the catalog's
    schema/source.schema.json (strict, additionalProperties: false)."""
    return {
        "id": SOURCE_RECORD_ID,
        "title": "kevinqz/coreai-fabric",
        "type": "github_repository",
        "url": FABRIC_REPO_URL,
        "owner": "kevinqz",
        "repo": "coreai-fabric",
        "trust": "project_primary",
        "volatility": "medium",
        "last_checked": today(),
        "notes": "First-party conversion pipeline: recipes in, provenance-verified .aimodel out.",
    }


def validate_against_catalog_schemas(
    catalog_path: Path, model_entry: dict, artifact_entry: dict
) -> list[str]:
    """Validate generated entries against the catalog clone's own schemas.
    Returns aggregated error strings (empty = valid)."""
    errors: list[str] = []
    schema_dir = catalog_path / "schema"
    for label, schema_file, entry in (
        ("model", "model.schema.json", model_entry),
        ("artifact", "artifact.schema.json", artifact_entry),
    ):
        spath = schema_dir / schema_file
        if not spath.is_file():
            errors.append(f"{label}: catalog schema missing at {spath}")
            continue
        schema = json.loads(spath.read_text())
        validator = Draft202012Validator(schema)
        for e in sorted(validator.iter_errors(entry), key=lambda e: list(e.absolute_path)):
            path = ".".join(str(p) for p in e.absolute_path) or "<root>"
            errors.append(f"{label}.{path}: {e.message}")
    # Actionable contract hint when the clone predates the shared field contract.
    if any("'fabric'" in e or "fabric" in e and "enum" in e for e in errors):
        errors.append(
            "hint: the catalog clone's schemas may predate the fabric field "
            "contract (source_group 'fabric', optional github, huggingface "
            "revision/files, provenance). Update the clone."
        )
    return errors


def _resolve_published_digests(recipe: Recipe) -> list[dict]:
    from . import hf

    published = recipe.data.get("published")
    if not published:
        raise SystemExit(
            f"error: recipe {recipe.id} is not published (status: {recipe.status}) — "
            f"run `coreai-fabric publish {recipe.id}` first."
        )
    try:
        return hf.file_digests(published["hf_repo"], published["revision"])
    except hf.HFError as exc:
        raise SystemExit(f"error: cannot fetch published file digests: {exc}") from exc


def _tool_version_from_manifest(root: Path, recipe: Recipe) -> str | None:
    mpath = manifest_path(root, recipe)
    if mpath.is_file():
        return json.loads(mpath.read_text()).get("tool_version")
    return None


def _notes_suffix_from_report(root: Path, recipe: Recipe) -> str:
    rpath = parity_report_path(root, recipe)
    if not rpath.is_file():
        return ""
    report = json.loads(rpath.read_text())
    return (
        f" Parity: gate A {report['gate_a']['status']}, "
        f"gate B {report['gate_b']['status']}."
    )


def cmd_register(args) -> int:
    root = find_root()
    recipe = find_recipe(args.id, root)

    # --mark-merged: the human merged the catalog PR; flip published -> registered.
    if getattr(args, "mark_merged", False):
        pr = recipe.data.get("catalog_pr")
        if not pr:
            err(f"recipe {recipe.id} has no catalog_pr — nothing to mark merged "
                "(run register to open the PR first)")
            return 1
        recipe.data["status"] = "registered"
        write_yaml(recipe.path, recipe.data)
        ok(f"recipe {recipe.id} status -> registered (PR {pr} merged)")
        return 0

    files = _resolve_published_digests(recipe)
    tool_version = _tool_version_from_manifest(root, recipe)
    model_entry = build_model_entry(
        recipe, files, notes_suffix=_notes_suffix_from_report(root, recipe)
    )
    artifact_entry = build_artifact_entry(recipe, files, tool_version)
    source_record = build_source_record()

    catalog_path = Path(args.catalog_path).resolve() if args.catalog_path else None
    if catalog_path:
        errors = validate_against_catalog_schemas(catalog_path, model_entry, artifact_entry)
        if errors:
            err(f"generated entries do not validate against {catalog_path}/schema:")
            for line in errors:
                print(f"  - {line}", file=sys.stderr)
            return 1
        ok("generated entries validate against the catalog schemas")

    if args.dry_run or not catalog_path:
        if not catalog_path:
            warn("no --catalog-path given: schema validation skipped, printing YAML only")
        print("# --- append to catalog.yaml under models: ---")
        print(dump_yaml({"models": [model_entry]}))
        print("# --- append to artifacts.yaml under artifacts: (and bump metadata.count) ---")
        print(dump_yaml({"artifacts": [artifact_entry]}))
        print("# --- ensure this record exists in sources.yaml under sources: ---")
        print(dump_yaml({"sources": [source_record]}))
        return 0

    return _apply_and_open_pr(root, recipe, catalog_path, model_entry, artifact_entry, source_record, args)


def _apply_and_open_pr(
    root: Path,
    recipe: Recipe,
    catalog_path: Path,
    model_entry: dict,
    artifact_entry: dict,
    source_record: dict,
    args,
) -> int:
    from . import CATALOG_REPO

    branch = f"fabric/add-{recipe.id}"

    def git(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(catalog_path), *cmd],
                              capture_output=True, text=True, check=check)

    status = git("status", "--porcelain").stdout.strip()
    if status:
        err(f"catalog clone at {catalog_path} has uncommitted changes — refusing to branch over them")
        return 1

    git("checkout", "-b", branch)

    # Text-append entries (preserves existing formatting/diff hygiene; the
    # catalog files end with their entry lists).
    _append_entry(catalog_path / "catalog.yaml", {"models": [model_entry]})
    _append_entry(catalog_path / "artifacts.yaml", {"artifacts": [artifact_entry]})
    _bump_artifact_count(catalog_path / "artifacts.yaml")
    _ensure_source_record(catalog_path / "sources.yaml", source_record)

    # Replay the catalog's OWN CI gate locally so the PR arrives green. Mirrors
    # .github/workflows/validate.yml, not a 3-script subset — otherwise a PR can
    # pass here and still fail catalog CI. Missing scripts (older checkout) are
    # skipped with a warning.
    gates = [
        ([sys.executable, "scripts/validate.py"], "validate", "scripts/validate.py"),
        ([sys.executable, "scripts/audit.py"], "audit", "scripts/audit.py"),
        ([sys.executable, "scripts/generate.py"], "generate", "scripts/generate.py"),
        ([sys.executable, "scripts/doc_test.py"], "doc_test", "scripts/doc_test.py"),
        ([sys.executable, "scripts/generate_templates.py", "--check"], "templates --check", "scripts/generate_templates.py"),
        ([sys.executable, "scripts/injection_lint.py"], "injection_lint", "scripts/injection_lint.py"),
        ([sys.executable, "-m", "pytest", "tests/test_p1_iocontract.py", "-q"], "io_contract tests", "tests/test_p1_iocontract.py"),
    ]
    for argv, label, needs in gates:
        if not (catalog_path / needs).exists():
            warn(f"catalog has no {needs}; skipping gate '{label}'")
            continue
        proc = subprocess.run(argv, cwd=catalog_path, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-6:])
            err(f"catalog gate '{label}' failed:\n{tail}")
            err(f"branch {branch} left in place at {catalog_path} for inspection; no PR opened")
            return 1
        print(f"  catalog gate {label}: ok")

    git("add", "-A")
    git("commit", "-m", f"feat: add {recipe.id} via coreai-fabric\n\nRecipe: {recipe_url(recipe)}")

    # Third-party publishers have no push access to the canonical catalog. Try a
    # direct push; on failure, fork and push to the fork, then open a cross-repo PR.
    head_ref = branch
    push = git("push", "-u", "origin", branch, check=False)
    if push.returncode != 0:
        warn("no push access to the catalog's origin — forking (third-party publisher path)")
        subprocess.run(["gh", "repo", "fork", CATALOG_REPO, "--remote=false", "--clone=false"],
                       cwd=catalog_path, capture_output=True, text=True)
        who = subprocess.run(["gh", "api", "user", "--jq", ".login"],
                             capture_output=True, text=True)
        fork_owner = who.stdout.strip()
        if not fork_owner:
            err(f"could not resolve your GitHub login for the fork:\n{who.stderr}")
            return 1
        fork_url = f"https://github.com/{fork_owner}/{CATALOG_REPO.split('/')[-1]}.git"
        git("remote", "add", "fork", fork_url, check=False)
        fpush = git("push", "-u", "fork", branch, check=False)
        if fpush.returncode != 0:
            err(f"push to fork {fork_owner} failed:\n{fpush.stderr}")
            return 1
        head_ref = f"{fork_owner}:{branch}"

    pr = subprocess.run(
        [
            "gh", "pr", "create",
            "--repo", CATALOG_REPO,
            "--head", head_ref,
            "--title", f"Add {recipe.id} (fabric conversion)",
            "--body", _pr_body(recipe),
        ],
        cwd=catalog_path, capture_output=True, text=True,
    )
    if pr.returncode != 0:
        err(f"gh pr create failed:\n{pr.stderr}")
        return 1
    pr_url = pr.stdout.strip()
    print(pr_url)

    # Honest status: the PR is OPEN, not merged. Indexing happens on merge, so
    # status stays `published` and we record the PR URL. `register --mark-merged`
    # flips to `registered` once a human merges.
    recipe.data["catalog_pr"] = pr_url
    write_yaml(recipe.path, recipe.data)
    ok(f"catalog PR opened: {pr_url}")
    ok(f"recipe {recipe.id} stays status='published' until the PR merges "
       f"(then: coreai-fabric register --mark-merged {recipe.id})")
    return 0


def _append_entry(path: Path, wrapper: dict) -> None:
    # Dump the single-entry wrapper and strip the top-level key line, leaving
    # a correctly-indented "- id: ..." list item to append.
    text = dump_yaml(wrapper)
    lines = text.splitlines()
    entry_text = "\n".join(lines[1:]) + "\n"
    existing = path.read_text()
    if not existing.endswith("\n"):
        existing += "\n"
    path.write_text(existing + entry_text)


def _bump_artifact_count(path: Path) -> None:
    text = path.read_text()
    match = re.search(r"^(\s*count:\s*)(\d+)\s*$", text, flags=re.MULTILINE)
    if not match:
        warn("artifacts.yaml has no metadata count: line to bump")
        return
    new = f"{match.group(1)}{int(match.group(2)) + 1}"
    path.write_text(text[: match.start()] + new + text[match.end():])


def _ensure_source_record(path: Path, record: dict) -> None:
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    ids = {s.get("id") for s in data.get("sources", [])}
    if record["id"] in ids:
        return
    _append_entry(path, {"sources": [record]})


def _pr_body(recipe: Recipe) -> str:
    published = recipe.data["published"]
    return (
        f"Adds `{recipe.id}`, converted and published via coreai-fabric.\n\n"
        f"- Recipe: {recipe_url(recipe)}\n"
        f"- Upstream: https://huggingface.co/{recipe.data['upstream']['hf_repo']}"
        f" (license: {recipe.data['upstream']['license']})\n"
        f"- Published artifact: https://huggingface.co/{published['hf_repo']}"
        f" @ `{published['revision']}`\n"
        f"- Provenance: `source_group: fabric`, `provenance.recipe_source: fabric`,"
        f" huggingface-only artifact block with pinned revision + per-file sha256\n\n"
        "Generated by `coreai-fabric register`. Parity and conversion reports "
        "ship inside the published HF repo.\n\n"
        "🤖 Generated with [coreai-fabric](https://github.com/kevinqz/coreai-fabric)\n"
    )
