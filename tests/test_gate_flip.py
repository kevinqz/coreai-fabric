"""RFC Phase 0d (F7 rule 1): the same-commit gate-flip guard."""
from __future__ import annotations

from scripts.check_gate_flip import _analyze_diff


def test_clean_status_bump_does_not_trip():
    # A pure status advance with NO gate change is fine. NOTE: `status:` is a
    # COLUMN-0 top-level key in every real recipe — the fixture must reflect that
    # (an indented `status:` masked the F7a regression the audit caught).
    diff = """\
diff --git a/recipes/x.yaml b/recipes/x.yaml
@@
-status: converted
+status: verified
"""
    gate, status, _ = _analyze_diff(diff)
    assert gate is False
    assert status is True  # status changed, but no gate redefine -> not a violation at the file level


def test_column0_status_flip_is_detected():
    # Regression guard for F7a: a top-level (column-0) status flip MUST be seen.
    diff = "@@\n-status: converted\n+status: verified\n"
    _, status, _ = _analyze_diff(diff)
    assert status is True


def test_gate_change_without_status_flip_does_not_trip():
    diff = """\
@@
 parity:
   gate_b:
-    threshold: 0.999
+    threshold: 0.99
-    tolerance: 0.0005
+    tolerance: 0.005
"""
    gate, status, _ = _analyze_diff(diff)
    assert gate is True
    assert status is False  # gate relaxed but no status flip -> caller sees (True, False) -> OK


def test_same_commit_gate_relax_and_status_flip_trips():
    # THE violation: relax the gate AND flip to a passing status in one diff.
    # Realistic layout: threshold is INDENTED under gate_b; status is COLUMN-0.
    diff = """\
@@
 parity:
   gate_b:
-    threshold: 0.999
+    threshold: 0.90
@@
-status: converted
+status: verified
"""
    gate, status, evidence = _analyze_diff(diff)
    assert gate is True
    assert status is True
    assert any("threshold" in e for e in evidence)
    assert any("status -> verified" in e for e in evidence)


def test_metric_change_counts_as_gate_redefinition():
    diff = """\
@@
   gate_b:
-    metric: greedy_parity
+    metric: benchmark_accuracy
@@
-status: draft
+status: converted
"""
    gate, status, _ = _analyze_diff(diff)
    assert gate is True
    assert status is True
