"""Audit fixes H1 (F6) + M1 (F2): the measured Gate-B number is DURABLE in the
recipe (survives a fresh clone → reaches the catalog + scorecard), and a measured
recipe is required to carry its full protocol signature."""
from __future__ import annotations

from pathlib import Path

from coreai_fabric.recipes import Recipe, _check_protocol_signature
from coreai_fabric.register import _catalog_evaluation, _load_parity_report
from coreai_fabric.verify import measurement_from_report


def _recipe(gate_b: dict) -> Recipe:
    return Recipe(Path("recipes/x.yaml"), {"id": "x", "parity": {"gate_b": gate_b}})


# ---- H1: measurement_from_report distills a durable, privacy-scrubbed core ----

def test_measurement_distills_numeric_core_and_scrubs_hardware():
    gb = {"metric": "graph_output_cosine", "status": "passed", "value": 0.99948,
          "min_action_cosine": 0.99948, "n_obs": 8,
          "environment": {"accelerator": "ANE", "chip": "M4 Max (private)", "os": "26.5"}}
    m = measurement_from_report(gb)
    assert m["value"] == 0.99948
    assert m["min_chunk_cosine"] == 0.99948   # aliased to the catalog-accepted name
    assert m["measured_on"] == "ANE"          # platform family only
    assert "chip" not in m and "os" not in m  # PRIVACY: never the specific build


def test_measurement_none_without_numeric_core():
    assert measurement_from_report({"metric": "x", "status": "not_run"}) is None


def test_register_reconstructs_number_on_fresh_clone(tmp_path):
    # H1/F6: build/<id>/parity-report.json is gitignored and absent on a fresh
    # clone. The recipe's durable `measured` block must let register rebuild a
    # report so the catalog gets the number (before the fix: catalog got None).
    recipe = _recipe({
        "metric": "action_parity", "threshold": 0.999, "tolerance": 0.001,
        "measured": {"status": "passed", "value": 0.9999, "min_chunk_cosine": 0.9999, "n_obs": 8},
        "protocol": {"n_obs": 8, "reference_dtype": "float32", "granularity": "per_row",
                     "input_protocol": "recorded", "graph_boundary": "encode -> denoise"},
    })
    report = _load_parity_report(tmp_path, recipe)   # tmp_path has no build/ → fresh clone
    assert report is not None
    ev = _catalog_evaluation(report)
    assert ev is not None, "the catalog must get a number from the durable recipe measurement"
    assert ev["min_chunk_cosine"] == 0.9999
    assert ev["n_obs"] == 8


# ---- M1: protocol signature required-when-measured, legacy exempt ----

def test_m1_measured_without_protocol_is_error():
    issues = _check_protocol_signature(_recipe(
        {"metric": "action_parity", "measured": {"value": 0.999, "status": "passed"}}))
    assert issues and issues[0].severity == "error"
    assert "protocol" in issues[0].path


def test_m1_measured_with_full_protocol_is_clean():
    issues = _check_protocol_signature(_recipe({
        "metric": "action_parity", "measured": {"value": 0.999, "status": "passed"},
        "protocol": {"input_protocol": "recorded", "reference_dtype": "float32",
                     "granularity": "per_row", "graph_boundary": "encode -> denoise"}}))
    assert issues == []


def test_m1_partial_protocol_is_error():
    issues = _check_protocol_signature(_recipe({
        "metric": "action_parity", "measured": {"value": 0.999, "status": "passed"},
        "protocol": {"granularity": "per_row"}}))   # missing 3 fields
    assert issues and issues[0].severity == "error"
    assert "input_protocol" in issues[0].message


def test_m1_legacy_unmeasured_recipe_is_exempt():
    # A pre-Phase-0 recipe has no `measured` block (its numbers lived in build/).
    # It must NOT be retroactively reddened.
    assert _check_protocol_signature(_recipe({"metric": "action_parity"})) == []
    assert _check_protocol_signature(_recipe(
        {"metric": "action_parity", "protocol": {"granularity": "per_row"}})) == []
