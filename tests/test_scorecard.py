"""RFC Phase 0e (F2/F6): the generated scorecard is reproducible from the recipes
and never invents a cross-cell ranking."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_committed_scorecard_matches_fresh_generation():
    # F14/L3: --check compares the COMMITTED file against a fresh generation WITHOUT
    # overwriting it first (the old test overwrote-then-checked — a tautology that
    # never guarded committed drift). Body-only (freshness stamp excluded) so it is
    # stable off the author's toolchain. Because the measured numbers are durable in
    # the recipes (H1/F6), this reproduces on a fresh clone with no build/.
    chk = subprocess.run(
        [sys.executable, "scripts/generate_scorecard.py", "--check"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert chk.returncode == 0, f"committed scorecard is stale vs fresh generation: {chk.stderr}"


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
