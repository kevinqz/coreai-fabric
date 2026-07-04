"""Gate A is implementable now and must be honest: pass, fail, and
skip-is-not-pass semantics; Gate B never runs without a configured runner."""
from __future__ import annotations

import json
from pathlib import Path

from coreai_fabric.recipes import Recipe
from coreai_fabric.verify import run_gate_a, run_gate_b


def _recipe(tmp_path: Path, expected_files=None, format_version=None) -> Recipe:
    expected: dict = {"bundle_files": expected_files or ["metadata.json"]}
    if format_version:
        expected["format_version"] = format_version
    return Recipe(
        path=tmp_path / "recipes" / "x.yaml",
        data={
            "id": "x",
            "upstream": {"hf_repo": "o/m", "license": "mit", "license_terms": "permissive"},
            "conversion": {"tool": "coreai-fabric-llm-export", "quantization": "none",
                           "precision": "float16"},
            "expected": expected,
            "parity": {
                "gate_a": {"checks": ["bundle_files_present", "metadata_json_parses",
                                      "metadata_matches_recipe"]},
                "gate_b": {"metric": "graph_output_cosine", "threshold": 0.999,
                           "tolerance": 0.0005},
            },
            "publish": {"hf_target_namespace": "o", "repo_name": "m-coreai"},
            "status": "converted",
        },
    )


def _make_bundle(tmp_path: Path, metadata: dict | str | None) -> Path:
    bundle = tmp_path / "build" / "x" / "x.aimodel"
    bundle.mkdir(parents=True)
    if metadata is not None:
        content = metadata if isinstance(metadata, str) else json.dumps(metadata)
        (bundle / "metadata.json").write_text(content)
    return bundle


def test_gate_a_passes_on_conforming_bundle(tmp_path):
    _make_bundle(tmp_path, {"format_version": "1"})
    result = run_gate_a(tmp_path, _recipe(tmp_path, format_version="1"))
    assert result["status"] == "passed"
    by_name = {c["name"]: c["status"] for c in result["checks"]}
    assert by_name["metadata_matches_recipe"] == "passed"


def test_gate_a_fails_when_bundle_missing(tmp_path):
    result = run_gate_a(tmp_path, _recipe(tmp_path))
    assert result["status"] == "failed"


def test_gate_a_fails_on_missing_expected_file(tmp_path):
    _make_bundle(tmp_path, {"a": 1})
    recipe = _recipe(tmp_path, expected_files=["metadata.json", "tokenizer.json"])
    result = run_gate_a(tmp_path, recipe)
    assert result["status"] == "failed"
    assert any("tokenizer.json" in c["detail"] for c in result["checks"])


def test_gate_a_fails_on_unparseable_metadata(tmp_path):
    _make_bundle(tmp_path, "{not json")
    result = run_gate_a(tmp_path, _recipe(tmp_path))
    assert result["status"] == "failed"


def test_gate_a_fails_on_format_version_mismatch(tmp_path):
    _make_bundle(tmp_path, {"format_version": "2"})
    result = run_gate_a(tmp_path, _recipe(tmp_path, format_version="1"))
    assert result["status"] == "failed"


def test_gate_a_checks_real_asset_version_key(tmp_path):
    # Real bundles (verified: coreai-core 1.0.0b2 asset on macOS 26.6) spell
    # the format-version key `assetVersion`, e.g. "2.0".
    _make_bundle(tmp_path, {"assetVersion": "2.0", "producer": "coreai-core 1.0.0b2"})
    result = run_gate_a(tmp_path, _recipe(tmp_path, format_version="2.0"))
    by_name = {c["name"]: c["status"] for c in result["checks"]}
    assert by_name["metadata_matches_recipe"] == "passed"
    mismatch = run_gate_a(tmp_path, _recipe(tmp_path, format_version="1.0"))
    assert mismatch["status"] == "failed"


def test_gate_a_records_skip_when_nothing_comparable_not_a_pass(tmp_path):
    # Bundle metadata has no overlapping keys with recipe expectations:
    # the check must record skipped, not passed.
    _make_bundle(tmp_path, {"unrelated": True})
    result = run_gate_a(tmp_path, _recipe(tmp_path))
    by_name = {c["name"]: c["status"] for c in result["checks"]}
    assert by_name["metadata_matches_recipe"] == "skipped"
    assert result["status"] == "passed"  # skip does not fail the gate, but is visible


def test_gate_b_not_run_without_runner(tmp_path, monkeypatch):
    monkeypatch.delenv("COREAI_FABRIC_PARITY_RUNNER", raising=False)
    result = run_gate_b(tmp_path, _recipe(tmp_path))
    assert result["status"] == "not_run"
    assert result["value"] is None
    assert "macOS" in result["reason"]


def test_gate_b_benchmark_accuracy_blocked_upstream(tmp_path, monkeypatch):
    # A production recipe (metric benchmark_accuracy) must report not_run with
    # the eval-stub reason — and must NOT shell out to a runner even when one is
    # configured (fabric's runner can't score a stateful asset; faking a failure
    # would be dishonest).
    monkeypatch.setenv("COREAI_FABRIC_PARITY_RUNNER", "/definitely/not/a/real/runner")
    recipe = _recipe(tmp_path)
    recipe.data["parity"]["gate_b"]["metric"] = "benchmark_accuracy"
    result = run_gate_b(tmp_path, recipe)
    assert result["status"] == "not_run"
    assert result["value"] is None
    assert "coreai.llm.eval" in result["reason"]
    assert "coming soon" in result["reason"]


# ---- action_parity: the two-venv action lane records an on-hardware measurement ----

def _action_recipe(tmp_path: Path) -> Recipe:
    r = _recipe(tmp_path)
    r.data["parity"]["gate_b"] = {"metric": "action_parity", "threshold": 0.999,
                                  "tolerance": 0.001, "max_action_mae": 0.05}
    return r


def _write_measured(tmp_path: Path, payload: dict) -> None:
    d = tmp_path / "build" / "x"
    d.mkdir(parents=True, exist_ok=True)
    (d / "action-parity-measured.json").write_text(json.dumps(payload))


def test_gate_b_action_parity_not_run_without_measurement(tmp_path):
    # No harness output next to the bundle => honestly not_run (never a faked number).
    result = run_gate_b(tmp_path, _action_recipe(tmp_path))
    assert result["status"] == "not_run"
    assert result["value"] is None
    assert "fabric never fakes" in result["reason"]


def test_gate_b_action_parity_records_passing_measurement(tmp_path):
    _write_measured(tmp_path, {"metric": "action_parity", "value": 0.99999,
                               "max_per_dim_mae": 2.1e-07, "min_action_cosine": 0.99999})
    result = run_gate_b(tmp_path, _action_recipe(tmp_path))
    assert result["status"] == "passed"
    assert result["value"] == 0.99999
    assert result["measurement_source"] == "action-parity-measured.json"
    assert result["max_per_dim_mae"] == 2.1e-07   # harness fields recorded verbatim


def test_gate_b_action_parity_fails_when_mae_over_cap(tmp_path):
    # Cosine passes but per-dim MAE exceeds the recipe cap => failed (both gates bind).
    _write_measured(tmp_path, {"metric": "action_parity", "value": 0.9995,
                               "max_per_dim_mae": 0.5})
    result = run_gate_b(tmp_path, _action_recipe(tmp_path))
    assert result["status"] == "failed"


def test_gate_b_action_parity_fails_below_threshold(tmp_path):
    _write_measured(tmp_path, {"metric": "action_parity", "value": 0.90,
                               "max_per_dim_mae": 0.01})
    result = run_gate_b(tmp_path, _action_recipe(tmp_path))
    assert result["status"] == "failed"


def test_gate_b_action_parity_rejects_malformed_measurement(tmp_path):
    _write_measured(tmp_path, {"metric": "action_parity", "value": None})
    result = run_gate_b(tmp_path, _action_recipe(tmp_path))
    assert result["status"] == "not_run"
    assert "not a valid" in result["reason"]
