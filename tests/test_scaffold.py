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
from coreai_fabric.scaffold import _default_gate_b, cmd_new, parse_preset_compression

# Verbatim rows from `coreai.model.registry --list-models --type llm`
# (captured 2026-07-03). Used to unit-test the parser without the toolchain.
REGISTRY_DUMP = """\
SHORT_NAME                       PLATFORM   COMPRESSION                                 CTX  HF_ID
qwen2.5-1.5b-instruct            macOS      4bit                                      32768  Qwen/Qwen2.5-1.5B-Instruct
qwen3-0.6b                       macOS      4bit                                       8192  Qwen/Qwen3-0.6B
qwen3-4b                         macOS      4bit                                      40960  Qwen/Qwen3-4B
gpt-oss-20b                      macOS      none                                      32768  openai/gpt-oss-20b
qwen3-0.6b                       iOS        qwen3_0_6b_mixed_4bit_8bit.yaml            4096  Qwen/Qwen3-0.6B
"""

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_SCHEMA = json.loads((REPO_ROOT / "schema" / "recipe.schema.json").read_text())


def _run_new(tmp_path, monkeypatch, argv: list[str]) -> int:
    (tmp_path / "recipes").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(scaffold, "find_root", lambda: tmp_path)
    args = build_parser().parse_args(["new", *argv])
    return cmd_new(args)


def test_default_gate_b_production_is_greedy_parity():
    # The novice's scaffold must reach the RUNNABLE Gate B (greedy_parity), not
    # benchmark_accuracy (permanently not_run on Apple's stubbed evaluator).
    gb = _default_gate_b("text-generation", production=True)
    assert gb["metric"] == "greedy_parity"
    assert gb["metric"] != "benchmark_accuracy"  # must not scaffold a never-runs gate
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
    assert data["parity"]["gate_b"]["metric"] == "greedy_parity"


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


def test_parse_preset_compression_reads_real_value():
    # The production quantization must come from the preset, not the scaffolder
    # default — a 4bit preset advertised as `none` would misreport in the catalog.
    assert parse_preset_compression(REGISTRY_DUMP, "qwen3-4b", "macOS") == "4bit"
    assert parse_preset_compression(REGISTRY_DUMP, "qwen2.5-1.5b-instruct", "macOS") == "4bit"
    # gpt-oss-20b is genuinely uncompressed — `none` here is correct, not a bug.
    assert parse_preset_compression(REGISTRY_DUMP, "gpt-oss-20b", "macOS") == "none"
    # Platform matters: the iOS row is a different (mixed) preset.
    assert parse_preset_compression(REGISTRY_DUMP, "qwen3-0.6b", "iOS") != "4bit"
    # Absent preset -> None (caller warns, never guesses).
    assert parse_preset_compression(REGISTRY_DUMP, "not-a-model", "macOS") is None


def test_production_scaffold_resolves_real_quantization(tmp_path, monkeypatch):
    # When the registry tool IS available, the production scaffold must write
    # the resolved compression (4bit), never the misleading `none` default.
    monkeypatch.setattr(scaffold, "preset_compression", lambda name, platform: "4bit")
    _run_new(tmp_path, monkeypatch, [
        "Qwen/Qwen3-4B", "--offline", "--license", "apache-2.0",
        "--pipeline-tag", "text-generation",
        "--tool", "coreai.llm.export", "--apple-registry-name", "qwen3-4b",
    ])
    data = yaml.safe_load((tmp_path / "recipes" / "qwen3-4b.yaml").read_text())
    assert data["conversion"]["quantization"] == "4bit"


def test_namespace_guard_refuses_shared_org(tmp_path, monkeypatch):
    # Publishing into a shared org silently (Kevin is a member) is the footgun —
    # `new` must refuse coreai-community without the explicit --i-am-mirroring.
    rc = _run_new(tmp_path, monkeypatch, [
        "Qwen/Qwen3-0.6B", "--offline", "--license", "apache-2.0",
        "--namespace", "coreai-community",
    ])
    assert rc == 1
    assert not list((tmp_path / "recipes").glob("*.yaml"))


def test_namespace_defaults_to_logged_in_user(tmp_path, monkeypatch):
    # With no --namespace, resolve the caller's own HF user (never a shared org).
    monkeypatch.setattr(scaffold, "whoami", None, raising=False)
    import coreai_fabric.scaffold as sc
    monkeypatch.setattr("huggingface_hub.whoami", lambda: {"name": "kevinqz"}, raising=False)
    rc = _run_new(tmp_path, monkeypatch, [
        "Qwen/Qwen3-0.6B", "--offline", "--license", "apache-2.0",
        "--pipeline-tag", "text-generation",
    ])
    assert rc == 0
    data = yaml.safe_load((tmp_path / "recipes" / "qwen3-0.6b.yaml").read_text())
    assert data["publish"]["hf_target_namespace"] == "kevinqz"


def test_registry_name_rejected_without_production_tool(tmp_path, monkeypatch):
    # --apple-registry-name with the default fabric driver is a user error:
    # the flag only means something for Apple's coreai.llm.export.
    rc = _run_new(tmp_path, monkeypatch, [
        "Qwen/Qwen3-0.6B", "--offline", "--license", "apache-2.0",
        "--apple-registry-name", "qwen3-0.6b",  # default --tool is the fabric driver
    ])
    assert rc == 1
    assert not (tmp_path / "recipes" / "qwen3-0.6b.yaml").exists()
