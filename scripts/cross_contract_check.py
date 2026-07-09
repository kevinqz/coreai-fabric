#!/usr/bin/env python3
"""Cross-contract check: prove fabric's register output still satisfies the LIVE catalog.

This is how the two repos "converse deeply without depending": fabric imports
nothing from the catalog at runtime, but this check clones the catalog and runs
fabric's real `build_model_entry` / `build_artifact_entry` output through the
catalog's OWN validators and invariant tests. If the catalog evolves its schema
or test invariants in a way fabric's generator no longer satisfies, this fails —
turning silent drift into a red build in both directions.

Usage:
    python scripts/cross_contract_check.py --catalog-path /path/to/coreai-catalog
    python scripts/cross_contract_check.py            # clones the catalog into a tempdir

Exit 0 = fabric's generated entries pass the catalog's validate + audit + the
io_contract invariant tests. Exit 1 = drift detected (message says which gate).
Exit 2 = the check could NOT run its full gate set (pytest missing, so the
io_contract invariants were skipped) — an environment gap, explicitly NOT a
drift signal. CI must treat 2 differently from 1: 1 means fabric broke the
contract; 2 means part of the contract went unverified. (The "no injectable
recipes" case stays exit 0 — nothing new to verify is not a failure.)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

FABRIC_ROOT = Path(__file__).resolve().parent.parent
CATALOG_REPO = "https://github.com/kevinqz/coreai-catalog.git"


def _load_fabric_register():
    sys.path.insert(0, str(FABRIC_ROOT))
    from coreai_fabric import register as reg
    from coreai_fabric.recipes import load_all_recipes

    return reg, load_all_recipes


def build_entries_for_recipe(reg, recipe, eval_keys=None):
    """Simulate the publish→register handoff without a real upload."""
    r = recipe
    # A published block + a minimal file manifest is what `publish` would have written.
    published = r.data.get("published") or {
        "hf_repo": f"{r.data['publish']['hf_target_namespace']}/{r.data['publish']['repo_name']}",
        "revision": "0" * 40,
        "date": "2026-01-01",
    }
    r.data["published"] = published
    files = [
        {"path": f"{r.id}.aimodel/main.mlirb", "sha256": "a" * 64, "size_bytes": 1024},
        {"path": f"{r.id}.aimodel/main.hash", "sha256": "b" * 64, "size_bytes": 32},
        {"path": f"{r.id}.aimodel/metadata.json", "sha256": "c" * 64, "size_bytes": 105},
    ]
    model_entry = reg.build_model_entry(r, files, catalog_eval_keys=eval_keys)
    artifact_entry = reg.build_artifact_entry(r, files, "0.4.1")
    source_record = reg.build_source_record() if hasattr(reg, "build_source_record") else None
    return model_entry, artifact_entry, source_record


def _pytest_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("pytest") is not None


def run(cmd, cwd, label):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    ok = proc.returncode == 0
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-4:]
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}")
    if not ok:
        for line in tail:
            print(f"        {line}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog-path", type=Path, default=None,
                    help="Path to a coreai-catalog checkout. If omitted, the repo is cloned.")
    ap.add_argument("--keep", action="store_true", help="Keep the temp working copy.")
    args = ap.parse_args()

    reg, load_all_recipes = _load_fabric_register()
    recipes = load_all_recipes(FABRIC_ROOT)
    print(f"cross-contract check: {len(recipes)} recipe(s) vs the live catalog")

    tmp = Path(tempfile.mkdtemp(prefix="fabric-xcontract-"))
    try:
        if args.catalog_path:
            work = tmp / "catalog"
            shutil.copytree(args.catalog_path, work,
                            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
        else:
            work = tmp / "catalog"
            r = subprocess.run(["git", "clone", "--depth", "1", CATALOG_REPO, str(work)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print("ERROR cloning catalog:\n" + r.stderr)
                return 1

        catalog_yaml = yaml.safe_load((work / "catalog.yaml").read_text())
        artifacts_yaml = yaml.safe_load((work / "artifacts.yaml").read_text())
        existing_ids = {m["id"] for m in catalog_yaml["models"]}
        # Feature-detect which evaluation fields THIS catalog checkout accepts, so
        # the emitted entries match the schema whether or not the Phase-0 schema PR
        # has landed (RFC §4.3 — the wire-up is merge-order-decoupled).
        eval_keys = reg.catalog_evaluation_keys(work)

        injected = 0
        for recipe in recipes:
            if not recipe.data.get("catalog"):
                continue  # draft with no catalog block — nothing to register yet
            kind = recipe.data["catalog"].get("bundle_kind")
            if kind != "llm":
                # fabric can't fully publish a non-LLM bundle yet: the card
                # renderer refuses it (S2) and register emits no io_contract for
                # it (C4). Testing "if published, would it satisfy the catalog?"
                # for a kind that CANNOT be published would red-build on a known,
                # documented gap. Skip with a log so the gap stays visible.
                print(f"  [skip] {recipe.id}: bundle_kind '{kind}' not yet publishable by fabric")
                continue
            if recipe.id in existing_ids:
                continue  # already in the catalog; skip to avoid a duplicate-id error
            me, ae, src = build_entries_for_recipe(reg, recipe, eval_keys=eval_keys)
            catalog_yaml["models"].append(me)
            artifacts_yaml["artifacts"].append(ae)
            injected += 1

        if not injected:
            print("  (no injectable recipes — all present or draft-only; contract unverified)")
            return 0

        artifacts_yaml["metadata"]["count"] = len(artifacts_yaml["artifacts"])
        (work / "catalog.yaml").write_text(
            yaml.safe_dump(catalog_yaml, sort_keys=False, allow_unicode=True))
        (work / "artifacts.yaml").write_text(
            yaml.safe_dump(artifacts_yaml, sort_keys=False, allow_unicode=True))
        print(f"  injected {injected} fabric-generated model+artifact entr(y/ies)")

        ok = True
        ok &= run([sys.executable, "scripts/validate.py"], work, "catalog validate.py")
        ok &= run([sys.executable, "scripts/audit.py"], work, "catalog audit.py")
        # The io_contract invariants need pytest. A MISSING pytest is an
        # environment problem, not contract drift — exit 2 (could-not-run), never
        # 1 (drift), so "pytest not installed" is never misread as "fabric broke
        # the catalog contract".
        if not _pytest_available():
            print("  [SKIP] catalog io_contract invariants — pytest not installed here")
            if not ok:
                print("cross-contract DRIFT: fabric's register output no longer satisfies the catalog.")
                return 1
            print("cross-contract INCOMPLETE: validate + audit passed, but the io_contract "
                  "invariant tests could NOT run (pytest missing). NOT a drift signal (exit 2).")
            return 2
        ok &= run([sys.executable, "-m", "pytest", "tests/test_p1_iocontract.py", "-q"],
                  work, "catalog io_contract invariants (bundle_kind/min_os)")
        if ok:
            print("cross-contract OK: fabric's register output satisfies the live catalog.")
            return 0
        print("cross-contract DRIFT: fabric's register output no longer satisfies the catalog.")
        return 1
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
