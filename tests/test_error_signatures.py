"""RFC Phase 1a (F8): error-signature classification table."""
from __future__ import annotations

from coreai_fabric.error_signatures import classify, outcome, outcome_for


def test_classifies_ane_program_ceiling():
    assert classify("CompilerEngine ... appleneuralengine Program load failure (0x10004)") == "0x10004"
    assert classify("error: ANE program limit exceeded") == "0x10004"


def test_classifies_complex_dtype():
    assert classify("unsupported dtype complex128 in matmul") == "complex128"


def test_classifies_sdpa_scale_fold():
    assert classify("FoldMultiplyIntoSDPAScale: failed to legalize") == "FoldMultiplyIntoSDPAScale"


def test_classifies_oom():
    assert classify("RuntimeError: out of memory (OOM)") == "OOM"
    assert classify("jetsam: the process was killed") == "OOM"


def test_classifies_parity_below_threshold():
    assert classify("gate B failed: cosine 0.91 below threshold 0.999") == "parity_below_threshold"


def test_classifies_import_error():
    assert classify("ModuleNotFoundError: No module named 'coreai_torch'") == "import_error"


def test_unclassified_fallback():
    assert classify("some novel unrelated error message") == "unclassified"
    assert classify("") == "unclassified"


def test_outcome_exit_zero_is_converted():
    assert outcome(0, "") == ("converted", "ok")


def test_outcome_blocks_on_ane_ceiling():
    o, sig = outcome(1, "Program load failure (0x10004)")
    assert o == "blocked"
    assert sig == "0x10004"


def test_outcome_fails_on_dtype():
    o, sig = outcome(1, "unsupported dtype complex128")
    assert o == "failed"
    assert sig == "complex128"


def test_outcome_unclassified_is_failed_not_blocked():
    # An unknown failure is treated as a fixable attempt, not a silent block.
    o, sig = outcome(1, "novel gibberish error")
    assert sig == "unclassified"
    assert o == "failed"


def test_outcome_for_known_signature():
    assert outcome_for("0x10004") == "blocked"
    assert outcome_for("parity_below_threshold") == "parity_below_threshold"
