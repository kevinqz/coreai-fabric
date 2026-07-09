"""RFC F16/F10 (audit L1/L5): the smoke battery is technique-keyed (so index-only
recipes are excluded by construction), every smoke states its exclusions, and the
torch-only SDPA smoke actually runs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKES = sorted((REPO_ROOT / "smokes").glob("*.py"))


def test_smokes_exist():
    names = {p.name for p in SMOKES}
    # The three techniques the playbook + RFC §5.3 name.
    assert {"moe-dense-fusion-lowering.py", "graph-split-chaining.py", "sdpa-mask-forms.py"} <= names


def test_smokes_are_technique_keyed_not_recipe_keyed():
    # F10: a smoke references a TECHNIQUE, never a recipe id / upstream — so an
    # index-only recipe (restricted upstream fabric never redistributes) can't be
    # pulled into the battery. Explicit guard, not just design-implicit.
    recipe_ids = {p.stem for p in (REPO_ROOT / "recipes").glob("*.yaml")}
    for s in SMOKES:
        text = s.read_text()
        hits = sorted(rid for rid in recipe_ids if rid in text)
        assert not hits, f"{s.name} references recipe id(s) {hits} — smokes must be technique-keyed (F10)"


def test_every_smoke_states_its_exclusions():
    # F12/F16: a smoke that doesn't say what it CANNOT see is incomplete by contract.
    for s in SMOKES:
        text = s.read_text().lower()
        assert "exclusion" in text or "cannot see" in text or "not a" in text, \
            f"{s.name} is missing its exclusion header (F16)"


def test_sdpa_smoke_passes():
    # Torch-only, runs on CI (no Apple toolchain). The bool keep-mask must be a
    # numeric no-op vs mask-free SDPA (playbook T2).
    out = subprocess.run([sys.executable, "smokes/sdpa-mask-forms.py"],
                         capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert out.returncode == 0, f"sdpa smoke failed:\n{out.stdout}\n{out.stderr}"
    assert "PASS" in out.stdout or "SKIP" in out.stdout
