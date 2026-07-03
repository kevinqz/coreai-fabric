"""`coreai-fabric verify` — Gate A (structure) + Gate B (numeric parity).

Gate A runs anywhere: bundle structure + metadata.json sanity vs recipe
expectations. Implemented and tested.

Gate B is numeric parity vs the upstream model (cosine thresholds from the
recipe). The protocol is defined in docs/parity-protocol.md; execution
requires macOS with the Apple Core AI runtime, driven by a Swift runner that
does not exist yet. This module shells out to a runner if one is configured
(COREAI_FABRIC_PARITY_RUNNER) and otherwise records gate_b.status: not_run —
it never fakes a pass.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .convert import bundle_path, manifest_path
from .recipes import Recipe, find_recipe
from .util import err, find_root, ok, utc_now_iso, warn

#: metadata.json keys that, when present, are cross-checked against the recipe.
#: Only keys observed in .aimodel bundles are checked; absent keys are recorded
#: as skipped, never assumed.
METADATA_RECIPE_KEYS = {
    "format_version": ("expected", "format_version"),
}


def parity_report_path(root: Path, recipe: Recipe) -> Path:
    return root / "build" / recipe.id / "parity-report.json"


def run_gate_a(root: Path, recipe: Recipe) -> dict:
    checks_requested = recipe.data["parity"]["gate_a"]["checks"]
    bundle = bundle_path(root, recipe)
    results: list[dict] = []

    def record(name: str, status: str, detail: str) -> None:
        results.append({"name": name, "status": status, "detail": detail})

    if not bundle.is_dir():
        record(
            "bundle_exists",
            "failed",
            f"{bundle.relative_to(root)} is not a directory — run `coreai-fabric convert {recipe.id}` first",
        )
        return {"status": "failed", "checks": results}
    record("bundle_exists", "passed", str(bundle.relative_to(root)))

    metadata: dict | None = None
    for check in checks_requested:
        if check == "bundle_files_present":
            missing = [
                rel for rel in recipe.data["expected"]["bundle_files"]
                if not (bundle / rel).exists()
            ]
            if missing:
                record(check, "failed", f"missing from bundle: {', '.join(missing)}")
            else:
                record(check, "passed", f"{len(recipe.data['expected']['bundle_files'])} expected file(s) present")
        elif check == "metadata_json_parses":
            meta_file = bundle / "metadata.json"
            if not meta_file.is_file():
                record(check, "failed", "metadata.json not found in bundle")
                continue
            try:
                loaded = json.loads(meta_file.read_text())
            except json.JSONDecodeError as exc:
                record(check, "failed", f"metadata.json is not valid JSON: {exc}")
                continue
            if not isinstance(loaded, dict):
                record(check, "failed", "metadata.json parses but is not a JSON object")
                continue
            metadata = loaded
            record(check, "passed", f"metadata.json parses ({len(metadata)} top-level keys)")
        elif check == "metadata_matches_recipe":
            if metadata is None:
                record(check, "skipped", "requires metadata_json_parses to pass first")
                continue
            mismatches = []
            compared = 0
            for meta_key, recipe_path in METADATA_RECIPE_KEYS.items():
                expected_val = recipe.data.get(recipe_path[0], {}).get(recipe_path[1])
                if expected_val is None or meta_key not in metadata:
                    continue
                compared += 1
                if str(metadata[meta_key]) != str(expected_val):
                    mismatches.append(
                        f"{meta_key}: bundle says {metadata[meta_key]!r}, recipe expects {expected_val!r}"
                    )
            if mismatches:
                record(check, "failed", "; ".join(mismatches))
            elif compared == 0:
                record(
                    check,
                    "skipped",
                    "no overlapping keys between recipe expectations and bundle metadata "
                    "(nothing to compare — this is not a pass)",
                )
            else:
                record(check, "passed", f"{compared} key(s) match")

    failed = any(c["status"] == "failed" for c in results)
    return {"status": "failed" if failed else "passed", "checks": results}


def run_gate_b(root: Path, recipe: Recipe) -> dict:
    """Numeric parity vs upstream. Shells to a Swift runner when configured;
    honestly reports not_run otherwise. See docs/parity-protocol.md."""
    gate = recipe.data["parity"]["gate_b"]
    base = {
        "metric": gate["metric"],
        "threshold": gate["threshold"],
        "tolerance": gate["tolerance"],
        "value": None,
    }
    if gate.get("greedy_token_exact") is not None:
        base["greedy_token_exact_required"] = gate["greedy_token_exact"]

    runner = os.environ.get("COREAI_FABRIC_PARITY_RUNNER")
    if not runner:
        return {
            **base,
            "status": "not_run",
            "reason": (
                "Gate B requires macOS + the Apple Core AI runtime driven by a "
                "Swift parity runner. No runner is configured "
                "(set COREAI_FABRIC_PARITY_RUNNER=/path/to/runner) and fabric "
                "does not ship one yet — see docs/parity-protocol.md for the "
                "protocol the runner must implement."
            ),
        }
    if not shutil.which(runner) and not Path(runner).is_file():
        return {
            **base,
            "status": "not_run",
            "reason": f"configured parity runner '{runner}' not found",
        }

    bundle = bundle_path(root, recipe)
    cmd = [
        runner,
        "--bundle", str(bundle),
        "--upstream", recipe.data["upstream"]["hf_repo"],
        "--metric", gate["metric"],
        "--threshold", str(gate["threshold"]),
        "--tolerance", str(gate["tolerance"]),
        "--report-json", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {
            **base,
            "status": "failed",
            "reason": f"parity runner exited {proc.returncode}: {proc.stderr.strip()[:500]}",
        }
    try:
        runner_report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            **base,
            "status": "failed",
            "reason": "parity runner did not emit valid JSON on stdout (see docs/parity-protocol.md)",
        }
    value = runner_report.get("value")
    passed = (
        isinstance(value, (int, float))
        and value >= gate["threshold"] - gate["tolerance"]
        and (not gate.get("greedy_token_exact") or runner_report.get("greedy_token_exact") is True)
    )
    result = {**base, "status": "passed" if passed else "failed", "value": value}
    if "greedy_token_exact" in runner_report:
        result["greedy_token_exact"] = runner_report["greedy_token_exact"]
    return result


def cmd_verify(args) -> int:
    root = find_root()
    recipe = find_recipe(args.id, root)

    gate_a = run_gate_a(root, recipe)
    gate_b = run_gate_b(root, recipe)

    if gate_a["status"] == "passed" and gate_b["status"] == "passed":
        overall = "passed"
    elif gate_a["status"] == "failed" or gate_b["status"] == "failed":
        overall = "failed"
    else:
        overall = "partial"

    report = {
        "recipe_id": recipe.id,
        "generated_at": utc_now_iso(),
        "bundle": str(bundle_path(root, recipe).relative_to(root)),
        "conversion_manifest": (
            str(manifest_path(root, recipe).relative_to(root))
            if manifest_path(root, recipe).is_file()
            else None
        ),
        "gate_a": gate_a,
        "gate_b": gate_b,
        "overall": overall,
    }
    rpath = parity_report_path(root, recipe)
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text(json.dumps(report, indent=2) + "\n")

    for check in gate_a["checks"]:
        marker = {"passed": "ok", "failed": "FAIL", "skipped": "skip"}[check["status"]]
        print(f"  gate A [{marker:4}] {check['name']}: {check['detail']}")
    print(f"  gate B [{gate_b['status']}] {gate_b['metric']}"
          + (f" = {gate_b['value']}" if gate_b.get("value") is not None else "")
          + (f" — {gate_b['reason']}" if gate_b.get("reason") else ""))
    print(f"report: {rpath.relative_to(root)} (overall: {overall})")

    if overall == "passed":
        from .util import write_yaml

        recipe.data["status"] = "verified"
        write_yaml(recipe.path, recipe.data)
        ok(f"recipe status -> verified ({recipe.path.name})")
        print(f"next: coreai-fabric publish {recipe.id}")
        return 0
    if overall == "partial":
        warn(
            "Gate A passed but Gate B did not run — the recipe stays at its "
            "current status. Numeric parity is required before verified."
        )
        return 2
    err("verification failed (see report)")
    return 1
