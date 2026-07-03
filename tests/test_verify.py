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
            "conversion": {"tool": "coreai-torch", "quantization": "none",
                           "precision": "fp16", "compute_units": "all"},
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
