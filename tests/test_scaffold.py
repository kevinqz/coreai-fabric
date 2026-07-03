"""`coreai-fabric new` scaffolding — the PRODUCTION path (coreai.llm.export +
--apple-registry-name) must scaffold an honest recipe, and the flag must be
guarded to that tool. Offline mode keeps these tests network-free."""
from __future__ import annotations

import yaml

from coreai_fabric import scaffold
from coreai_fabric.cli import build_parser
from coreai_fabric.scaffold import _default_gate_b, cmd_new


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


def test_registry_name_rejected_without_production_tool(tmp_path, monkeypatch):
    # --apple-registry-name with the default fabric driver is a user error:
    # the flag only means something for Apple's coreai.llm.export.
    rc = _run_new(tmp_path, monkeypatch, [
        "Qwen/Qwen3-0.6B", "--offline", "--license", "apache-2.0",
        "--apple-registry-name", "qwen3-0.6b",  # default --tool is the fabric driver
    ])
    assert rc == 1
    assert not (tmp_path / "recipes" / "qwen3-0.6b.yaml").exists()
