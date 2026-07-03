"""`coreai-fabric new` scaffolding — the PRODUCTION path (coreai.llm.export +
--apple-registry-name) must scaffold an honest recipe, and the flag must be
guarded to that tool. Offline mode keeps these tests network-free."""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from coreai_fabric import scaffold
from coreai_fabric.cli import build_parser
from coreai_fabric.scaffold import _default_gate_b, cmd_new

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_SCHEMA = json.loads((REPO_ROOT / "schema" / "recipe.schema.json").read_text())


def _run_new(tmp_path, monkeypatch, argv: list[str]) -> int:
    (tmp_path / "recipes").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(scaffold, "find_root", lambda: tmp_path)
    args = build_parser().parse_args(["new", *argv])
    return cmd_new(args)


def test_default_gate_b_production_is_benchmark_accuracy():
    gb = _default_gate_b("text-generation", production=True)
    assert gb["metric"] == "benchmark_accuracy"
    # No greedy_token_exact — that's a static-graph notion, not for a KV-cache asset.
    assert "greedy_token_exact" not in gb


def test_default_gate_b_static_llm_stays_logit_cosine():
    gb = _default_gate_b("text-generation", production=False)
    assert gb["metric"] == "per_token_logit_cosine"
    assert gb["greedy_token_exact"] is True


def test_production_scaffold_wires_registry_name_and_gate(tmp_path, monkeypatch):
    rc = _run_new(tmp_path, monkeypatch, [
        "Qwen/Qwen3-0.6B", "--offline", "--license", "apache-2.0",
        "--pipeline-tag", "text-generation",
        "--tool", "coreai.llm.export", "--apple-registry-name", "qwen3-0.6b",
    ])
    assert rc == 0
    data = yaml.safe_load((tmp_path / "recipes" / "qwen3-0.6b.yaml").read_text())
    conv = data["conversion"]
    assert conv["tool"] == "coreai.llm.export"
    assert conv["apple_registry_name"] == "qwen3-0.6b"
    assert data["parity"]["gate_b"]["metric"] == "benchmark_accuracy"


def test_scaffolded_production_recipe_is_schema_valid(tmp_path, monkeypatch):
    # Regression: the scaffolder must emit a recipe that validates against the
    # FULL schema — the catalog block needs bundle_kind + runtime_facts (boundary
    # redteam), not the pre-P1 runner/aot_required placeholders. A structurally
    # "written" recipe that fails schema breaks every downstream step.
    _run_new(tmp_path, monkeypatch, [
        "Qwen/Qwen3-0.6B", "--offline", "--license", "apache-2.0",
        "--pipeline-tag", "text-generation",
        "--tool", "coreai.llm.export", "--apple-registry-name", "qwen3-0.6b",
    ])
    data = yaml.safe_load((tmp_path / "recipes" / "qwen3-0.6b.yaml").read_text())
    errors = [e.message for e in Draft202012Validator(RECIPE_SCHEMA).iter_errors(data)]
    assert errors == [], f"scaffolded recipe fails schema: {errors}"
    cat = data["catalog"]
    assert cat["bundle_kind"] == "llm"
    assert cat["min_os"] == {"macos": "27.0", "ios": "27.0"}
    rf = cat["runtime_facts"]
    for field in ("stock_runtime", "custom_kernel", "patch_required", "aot_required"):
        assert isinstance(rf[field], bool)
    # No stale pre-P1 placeholders leaked in.
    assert "aot_required" not in cat  # it lives INSIDE runtime_facts now


def test_registry_name_rejected_without_production_tool(tmp_path, monkeypatch):
    # --apple-registry-name with the default fabric driver is a user error:
    # the flag only means something for Apple's coreai.llm.export.
    rc = _run_new(tmp_path, monkeypatch, [
        "Qwen/Qwen3-0.6B", "--offline", "--license", "apache-2.0",
        "--apple-registry-name", "qwen3-0.6b",  # default --tool is the fabric driver
    ])
    assert rc == 1
    assert not (tmp_path / "recipes" / "qwen3-0.6b.yaml").exists()
