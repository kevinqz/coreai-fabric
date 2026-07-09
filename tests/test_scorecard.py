"""RFC Phase 0e (F2/F6): the generated scorecard is reproducible from the recipes
and never invents a cross-cell ranking."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_scorecard_generator_runs_and_is_deterministic():
    # Generate twice into a temp and confirm identical output (no timestamps that
    # change per second would break this; verified_at is day-granularity).
    env = {"COREAI_FABRIC_ROOT": str(REPO_ROOT)}
    out1 = subprocess.run(
        [sys.executable, "scripts/generate_scorecard.py"], capture_output=True, text=True, env=env
    )
    assert out1.returncode == 0, out1.stderr
    text1 = (REPO_ROOT / "docs" / "scorecard.md").read_text()

    # Re-run: --check should now pass (the file matches a fresh generation).
    chk = subprocess.run(
        [sys.executable, "scripts/generate_scorecard.py", "--check"], capture_output=True, text=True
    )
    assert chk.returncode == 0, f"scorecard not reproducible: {chk.stderr}"


def test_scorecard_header_carries_freshness_stamps():
    # F15: every generated fact carries verified_at + toolchain_version.
    text = (REPO_ROOT / "docs" / "scorecard.md").read_text()
    assert "**verified_at:**" in text
    assert "**toolchain_version:**" in text


def test_scorecard_documents_no_cross_cell_ranking():
    text = (REPO_ROOT / "docs" / "scorecard.md").read_text()
    assert "no cross-cell" in text.lower()
    # Never claims a global "best".
    assert "best overall" not in text.lower()
