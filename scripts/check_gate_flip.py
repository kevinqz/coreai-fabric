#!/usr/bin/env python3
"""RFC Phase 0d (F7 rule 1): no same-commit gate flip.

A single commit that RELAXES the parity gate (changes `gate_b.metric` /
`gate_b.threshold` / `gate_b.tolerance`) AND simultaneously flips a recipe's
status toward a passing state (failed→verified, converted→verified,
draft→converted, blocked→verified, …) is a gaming move. The pi0fast precedent:
gate relaxations landed in the same unreviewed commit as the results they
enabled. This guard makes that class a CI failure.

It runs against `git diff <base>...HEAD` (default base = origin/main). On a
clean diff (no gate change, or a gate change without a status flip) it exits 0.
When run outside a git repo or with no diff range, it exits 0 (no-op) so the
guard never blocks unrelated work.

Usage:
    python scripts/check_gate_flip.py                 # origin/main...HEAD
    python scripts/check_gate_flip.py --base main     # main...HEAD
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# A status change toward a PASSING state. We flag any line that SETS status to a
# post-attempt passing state, regardless of the prior value, when it co-occurs
# with a gate relaxation. Verified is the gateway to publish, so it is the load-
# bearing flip; converted is flagged too (a draft→converted + gate relax lets a
# weaker gate ride into verify).
PASSING_STATUSES = ("converted", "verified", "published", "registered")

# The gate-definition fields a relaxation changes.
GATE_FIELDS = ("metric", "threshold", "tolerance")

# Match added YAML lines like `    threshold: 0.99` (indented, under gate_b) or
# `status: verified` (COLUMN 0 in every real recipe). The leading whitespace is
# OPTIONAL (`\s*`): `status:` is a top-level key, so a `\s+` here silently missed
# the exact flip this guard exists to catch (the F7a regression the audit found).
_FIELD_RE = re.compile(r"^\+?\s*(threshold|metric|tolerance|status):\s*(.+)$")


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout


def _diff_recipes(base: str) -> list[tuple[str, str]]:
    """[(path, diff_text)] for every recipes/*.yaml changed in base...HEAD."""
    names = _git("diff", "--name-only", f"{base}...HEAD").splitlines()
    recipe_names = [n for n in names if n.startswith("recipes/") and n.endswith(".yaml")]
    out = []
    for name in recipe_names:
        diff = _git("diff", f"{base}...HEAD", "--", name)
        if diff:
            out.append((name, diff))
    return out


def _analyze_diff(diff: str) -> tuple[bool, bool, list[str]]:
    """Return (gate_redefined, status_flipped_passing, evidence) for one file's diff.

    We only inspect ADDED (+) lines that touch gate_b fields or status. Context
    is captured by walking the diff hunks and tracking whether we are inside a
    `gate_b:` mapping (cheap structural heuristic on indentation)."""
    in_gate_b = False
    gate_redefined = False
    status_flipped = False
    evidence: list[str] = []

    for raw in diff.splitlines():
        # Track the mapping scope via context (non-+/-) lines and added lines.
        line_for_scope = raw.lstrip("+-@ ") if raw.startswith(("+", "-")) or raw.startswith("@") else raw
        stripped = line_for_scope.rstrip()
        if re.match(r"^\s*gate_b:\s*$", stripped):
            in_gate_b = True
            continue
        # A top-level key (no indent) ends the gate_b scope.
        if in_gate_b and re.match(r"^[A-Za-z_][A-Za-z0-9_]*:\s*$", stripped):
            in_gate_b = False

        if not raw.startswith("+") or raw.startswith("+++"):
            continue
        m = _FIELD_RE.match(raw)
        if not m:
            continue
        field, value = m.group(1), m.group(2).strip()
        if in_gate_b and field in GATE_FIELDS:
            gate_redefined = True
            evidence.append(f"gate_b.{field} -> {value}")
        elif field == "status" and value.strip("'\"") in PASSING_STATUSES:
            status_flipped = True
            evidence.append(f"status -> {value}")
    return gate_redefined, status_flipped, evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main",
                        help="git ref to diff against (default: origin/main)")
    args = parser.parse_args(argv)

    # Resolve base; if origin/main doesn't exist (e.g. shallow clone), fall back
    # to HEAD^ so a guard run never hard-fails on infra — a no-op is safer than
    # a false positive.
    if not _git("rev-parse", "--verify", args.base).strip():
        fallback = "HEAD^"
        if not _git("rev-parse", "--verify", fallback).strip():
            print(f"check_gate_flip: no git base to diff ({args.base} / {fallback} missing) — no-op")
            return 0
        base = fallback
    else:
        base = args.base

    violations = []
    for path, diff in _diff_recipes(base):
        gate, status, evidence = _analyze_diff(diff)
        if gate and status:
            violations.append((path, evidence))

    if not violations:
        print("check_gate_flip: no same-commit gate+status flips detected")
        return 0

    print("check_gate_flip: REFUSED — a gate redefinition and a passing status flip "
          "landed in the same diff (RFC F7 rule 1). Split them:\n", file=sys.stderr)
    for path, evidence in violations:
        print(f"  {path}:", file=sys.stderr)
        for e in evidence:
            print(f"    + {e}", file=sys.stderr)
    print("\nMove the gate change (threshold/metric/tolerance) into its OWN commit, "
          "land it, THEN flip the status in a separate commit that cites the "
          "re-measured result.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
