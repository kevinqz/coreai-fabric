"""RFC Phase 2 (F5/F14): optional catalog.blocks + the generated reverse index.
`used_by` is derived never stored; the index never says SOLVED; orphaned refs flag."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_schema_accepts_optional_blocks(tmp_path):
    """An optional catalog.blocks array validates (F5)."""
    import json
    from jsonschema import Draft202012Validator

    schema = json.loads((REPO_ROOT / "schema" / "recipe.schema.json").read_text())
    minimal = {
        "id": "x", "upstream": {"hf_repo": "o/m", "license": "mit", "license_terms": "permissive"},
        "conversion": {"tool": "t", "quantization": "none", "precision": "float16"},
        "expected": {"bundle_files": ["metadata.json"]},
        "parity": {"gate_a": {"checks": ["bundle_files_present"]},
                   "gate_b": {"metric": "graph_output_cosine", "threshold": 0.999, "tolerance": 0.0005}},
        "publish": {"hf_target_namespace": "o", "repo_name": "m"},
        "catalog": {"name": "X", "family": "f", "capabilities": ["text-generation"],
                    "modalities": {"input": ["text"], "output": ["text"]},
                    "bundle_kind": "llm",
                    "runtime_facts": {"stock_runtime": True, "custom_kernel": False,
                                      "patch_required": False, "aot_required": False},
                    "blocks": ["qwen2-lm", "siglip-so400m-vit"]},
        "status": "draft",
    }
    errors = list(Draft202012Validator(schema).iter_errors(minimal))
    assert errors == [], [e.message for e in errors]


def test_unknown_block_id_is_warning_not_error():
    """F5: an unknown block id warns, never errors (YAGNI — don't block conversion)."""
    from coreai_fabric.recipes import _check_blocks, Recipe

    r = Recipe(path=Path("recipes/x.yaml"),
               data={"catalog": {"blocks": ["qwen2-lm", "nonexistent-block"]}})
    issues = _check_blocks(r)
    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert "nonexistent-block" in issues[0].message


def test_known_block_id_produces_no_warning():
    from coreai_fabric.recipes import _check_blocks, Recipe

    r = Recipe(path=Path("recipes/x.yaml"),
               data={"catalog": {"blocks": ["qwen2-lm", "siglip-so400m-vit"]}})
    assert _check_blocks(r) == []


def test_committed_blocks_index_matches_fresh_generation():
    # F14/L3: --check the COMMITTED file against a fresh generation WITHOUT
    # overwriting it first (no overwrite-then-check tautology). Body-only, so it is
    # stable off the author's toolchain and reproduces on a fresh clone.
    chk = subprocess.run([sys.executable, "scripts/generate_blocks_index.py", "--check"],
                         capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert chk.returncode == 0, f"committed blocks-index is stale vs fresh generation: {chk.stderr}"


def test_blocks_index_never_claims_solved_as_a_status():
    # No measured-Gate-B table cell reads "SOLVED" (the honest vocabulary is
    # `measured @ {envelope}` or —). The word may appear in the explanatory prose
    # that says "NEVER SOLVED" — that's the rule itself, not a status.
    text = (REPO_ROOT / "docs" / "blocks-index.md").read_text()
    # A table status cell of "SOLVED" would appear as "| SOLVED |" or "| SOLVED".
    assert "| SOLVED" not in text and "SOLVED |" not in text
    # The honest vocabulary is present.
    assert "measured @" in text or "no recipe composes" in text


def test_blocks_index_carries_freshness_stamps():
    text = (REPO_ROOT / "docs" / "blocks-index.md").read_text()
    assert "**verified_at:**" in text
    assert "**toolchain_version:**" in text


def test_used_by_is_derived_not_stored():
    # The vocab file never stores a used_by list (F14: that reintroduces merge collisions).
    vocab = (REPO_ROOT / "schema" / "blocks-vocab.yaml").read_text()
    assert "used_by" not in vocab
