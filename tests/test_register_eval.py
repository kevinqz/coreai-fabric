"""RFC Phase 0c (F6/F7): the measured parity number + protocol signature reach
the catalog `evaluation`, and the fidelity tier is derived from margin/n_obs/
waivers — never a collapse of any pass to high_fidelity."""
from __future__ import annotations

from coreai_fabric.register import (
    _catalog_evaluation,
    _fidelity_tier,
)
from coreai_fabric.recipes import Recipe


def _report(gate_b: dict) -> dict:
    return {"gate_a": {"status": "passed"}, "gate_b": gate_b}


def test_eval_carries_numeric_value_and_min_cosine():
    # F6: the numeric value + min cosine reach the catalog (dropped before).
    gb = {"metric": "graph_output_cosine", "status": "passed", "value": 0.9999,
          "min_cosine": 0.9999, "n_obs": 8, "threshold": 0.999, "tolerance": 0.0005}
    ev = _catalog_evaluation(_report(gb))
    assert ev["value"] == 0.9999
    assert ev["min_cosine"] == 0.9999
    assert ev["n_obs"] == 8


def test_eval_normalizes_action_cosine_alias_to_min_cosine():
    # The action lane reports min as `min_action_cosine`; normalize to `min_cosine`.
    gb = {"metric": "action_parity", "status": "passed", "value": 0.999,
          "min_action_cosine": 0.999, "n_obs": 8}
    ev = _catalog_evaluation(_report(gb))
    assert ev["min_cosine"] == 0.999


def test_eval_surfaces_waivers_from_protocol():
    # F7 rule 3: waivers surface into the catalog evaluation, never silent.
    gb = {"metric": "action_parity", "status": "passed", "value": 0.999, "n_obs": 8,
          "protocol": {"waivers": ["near_zero_action"], "granularity": "per_row"}}
    ev = _catalog_evaluation(_report(gb))
    assert ev["waivers"] == ["near_zero_action"]
    assert ev["protocol"]["waivers"] == ["near_zero_action"]
    assert ev["protocol"]["granularity"] == "per_row"


def test_eval_carries_protocol_block():
    gb = {"metric": "graph_output_cosine", "status": "passed", "value": 0.9999, "n_obs": 8,
          "protocol": {"n_obs": 8, "reference_dtype": "float32", "granularity": "flattened",
                       "input_protocol": "recorded", "graph_boundary": "single-graph forward"}}
    ev = _catalog_evaluation(_report(gb))
    assert ev["protocol"]["reference_dtype"] == "float32"
    assert ev["protocol"]["granularity"] == "flattened"
    assert ev["protocol"]["input_protocol"] == "recorded"


def test_fidelity_tier_bare_pass_is_balanced_not_high_fidelity():
    # F7 rule 2: a pass right at the bar, or without n_obs, is NOT high_fidelity.
    gb = {"status": "passed", "value": 0.999, "threshold": 0.999}  # zero margin
    assert _fidelity_tier("none", _report(gb)) == "balanced"
    gb2 = {"status": "passed", "value": 1.0, "threshold": 0.9}  # margin ok but no n_obs
    assert _fidelity_tier("none", _report(gb2)) == "balanced"


def test_fidelity_tier_high_fidelity_needs_margin_and_sample():
    gb = {"status": "passed", "value": 0.9999, "threshold": 0.999, "n_obs": 8}  # margin 0.0009
    assert _fidelity_tier("none", _report(gb)) == "high_fidelity"


def test_fidelity_tier_waivered_pass_is_balanced():
    gb = {"status": "passed", "value": 1.0, "threshold": 0.9, "n_obs": 16,
          "protocol": {"waivers": ["near_zero_action"]}}
    assert _fidelity_tier("none", _report(gb)) == "balanced"


def test_fidelity_tier_failed_is_size():
    gb = {"status": "failed", "value": 0.8}
    assert _fidelity_tier("int8", _report(gb)) == "size"


def test_fidelity_tier_unmeasured_falls_back_to_quant_hint():
    # not_run with no measurement: the quant tier is an honest tiebreaker hint.
    assert _fidelity_tier("int8", None) == "high_fidelity"
    assert _fidelity_tier("int4", None) == "size"
    assert _fidelity_tier("none", None) is None


def test_eval_none_when_unmeasured():
    assert _catalog_evaluation(None) is None
    assert _catalog_evaluation({"gate_b": {"metric": "x", "status": "not_run"}}) is None
