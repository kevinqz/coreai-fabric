"""RFC Phase 0c + F13 (F6/F7): the measured parity number + protocol signature
reach the catalog via the vocabulary the LIVE catalog schema accepts, and the
fidelity tier is derived from margin/n_obs/waivers — never a collapse of any
pass to high_fidelity. The generic `value`/`protocol` fields live in the recipe
(catalog_protocol_extension) pending the batched catalog schema PR."""
from __future__ import annotations

from coreai_fabric.register import (
    _catalog_evaluation,
    _fidelity_tier,
    catalog_protocol_extension,
)


def _report(gate_b: dict) -> dict:
    return {"gate_a": {"status": "passed"}, "gate_b": gate_b}


def test_eval_carries_action_cosine_in_catalog_vocabulary():
    # F6: the numeric min cosine reaches the catalog under the action-lane name
    # the live catalog accepts (min_chunk_cosine), aliased from min_action_cosine.
    gb = {"metric": "action_parity", "status": "passed", "value": 0.999,
          "min_action_cosine": 0.999, "n_obs": 8}
    ev = _catalog_evaluation(_report(gb))
    assert ev["min_chunk_cosine"] == 0.999
    assert ev["n_obs"] == 8


def test_eval_carries_greedy_parity_numbers():
    gb = {"metric": "greedy_parity", "status": "passed", "value": 1.0,
          "margin_gated_match_rate": 1.0, "argmax_match_rate": 0.958,
          "matched": 46, "compared": 48, "reference_dtype": "float16"}
    ev = _catalog_evaluation(_report(gb))
    assert ev["argmax_match_rate"] == 0.958
    assert ev["margin_gated_match_rate"] == 1.0


def test_protocol_extension_carries_value_and_protocol():
    # F6/F2: the generic value + full protocol live here, pending the catalog PR.
    gb = {"metric": "graph_output_cosine", "status": "passed", "value": 0.9999,
          "min_cosine": 0.9999, "threshold": 0.999, "n_obs": 8,
          "protocol": {"n_obs": 8, "reference_dtype": "float32", "granularity": "flattened",
                       "input_protocol": "recorded", "graph_boundary": "single-graph forward"}}
    ext = catalog_protocol_extension(_report(gb))
    assert ext["value"] == 0.9999
    assert ext["min_cosine"] == 0.9999
    assert ext["protocol"]["reference_dtype"] == "float32"
    assert ext["protocol"]["granularity"] == "flattened"


def test_protocol_extension_carries_waivers():
    gb = {"metric": "action_parity", "status": "passed", "value": 0.999,
          "protocol": {"waivers": ["near_zero_action"], "granularity": "per_row"}}
    ext = catalog_protocol_extension(_report(gb))
    assert ext["protocol"]["waivers"] == ["near_zero_action"]


def test_eval_emits_no_rejected_additional_properties():
    # F13: the catalog uses additionalProperties:false. value/protocol must NOT
    # appear in the catalog-bound evaluation (they live in catalog_protocol_extension).
    gb = {"metric": "action_parity", "status": "passed", "value": 0.999,
          "min_action_cosine": 0.999, "n_obs": 8,
          "protocol": {"granularity": "per_row"}}
    ev = _catalog_evaluation(_report(gb))
    assert "value" not in ev
    assert "protocol" not in ev
    assert "min_cosine" not in ev  # normalized to min_chunk_cosine (catalog-accepted)


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


def test_waivers_surface_in_catalog_reason():
    # M2 (F7 rule 3): a waivered pass must not reach the catalog looking clean.
    # The live catalog schema has no `waivers` field, so they fold into `reason`.
    gb = {"metric": "action_parity", "status": "passed", "value": 0.999,
          "min_action_cosine": 0.999, "n_obs": 8, "reason": "chunk cosine ok",
          "protocol": {"waivers": ["near_zero_action"], "granularity": "per_row"}}
    ev = _catalog_evaluation(_report(gb))
    assert "near_zero_action" in ev["reason"]
    assert "waivers" not in ev            # folded into reason, not a rejected property


def test_no_waiver_leaves_reason_untouched():
    gb = {"metric": "action_parity", "status": "passed", "value": 0.999,
          "min_action_cosine": 0.999, "n_obs": 8, "reason": "clean"}
    ev = _catalog_evaluation(_report(gb))
    assert ev["reason"] == "clean"
