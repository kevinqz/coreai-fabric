"""convert adapter tests — the command layout is the one VERIFIED against
real tools (coreai-fabric-llm-export executed on real hardware 2026-07-03;
coreai.llm.export read from the apple/coreai-models source at tag 0.1.0)."""
from __future__ import annotations

from pathlib import Path

from coreai_fabric.convert import (
    LLM_EXPORT_TOOLS,
    REVISION_CAPABLE_TOOLS,
    build_command,
    bundle_path,
    is_script_tool,
    script_tool_hint,
)
from coreai_fabric.recipes import Recipe


def _recipe(tool: str = "coreai-fabric-llm-export", args: dict | None = None,
            revision: str | None = "a" * 40) -> Recipe:
    upstream = {"hf_repo": "Qwen/Qwen3-0.6B", "license": "apache-2.0",
                "license_terms": "permissive"}
    if revision:
        upstream["revision"] = revision
    return Recipe(
        path=Path("recipes/qwen3-0.6b.yaml"),
        data={
            "id": "qwen3-0.6b",
            "upstream": upstream,
            "conversion": {
                "tool": tool,
                "quantization": "none",
                "precision": "float16",
                "args": args if args is not None else {"platform": "macOS"},
            },
            "expected": {"bundle_files": ["metadata.json", "main.mlirb", "main.hash"]},
            "parity": {
                "gate_a": {"checks": ["bundle_files_present"]},
                "gate_b": {"metric": "per_token_logit_cosine", "threshold": 0.999,
                           "tolerance": 0.0005},
            },
            "publish": {"hf_target_namespace": "o", "repo_name": "r"},
            "status": "draft",
        },
    )


def test_build_command_verified_layout(tmp_path):
    recipe = _recipe()
    output = bundle_path(tmp_path, recipe)
    cmd = build_command(recipe, "coreai-fabric-llm-export", output)
    assert cmd[0] == "coreai-fabric-llm-export"
    assert cmd[1] == "Qwen/Qwen3-0.6B"  # positional model id — no `export` subcommand
    joined = " ".join(cmd)
    # Verified flag names (NOT the old assumed --precision/--quantization/--compute-units):
    assert "--compute-precision float16" in joined
    assert "--compression none" in joined
    assert f"--output-dir {tmp_path / 'build'}" in joined
    assert "--output-name qwen3-0.6b" in joined
    assert "--overwrite" in joined
    assert "--platform macOS" in joined
    assert "--compute-units" not in joined  # flag never existed in the real toolchain
    # Bundle lands exactly at fabric's expected path:
    # <output-dir>/<output-name>/<output-name>.aimodel == build/<id>/<id>.aimodel
    assert output == tmp_path / "build" / "qwen3-0.6b" / "qwen3-0.6b.aimodel"


def test_revision_only_passed_to_revision_capable_tools(tmp_path):
    recipe = _recipe()
    output = bundle_path(tmp_path, recipe)
    fabric_cmd = build_command(recipe, "coreai-fabric-llm-export", output)
    assert "--revision" in fabric_cmd  # fabric's driver pins upstream
    # Apple's coreai.llm.export has NO --revision flag (verified from source):
    apple_cmd = build_command(recipe, "coreai.llm.export", output)
    assert "--revision" not in apple_cmd


def test_coreai_llm_export_raw_id_adds_experimental(tmp_path):
    # A raw HF id (no apple_registry_name) needs --experimental for the real
    # coreai.llm.export, and keeps precision/quantization.
    recipe = _recipe(tool="coreai.llm.export")
    cmd = build_command(recipe, "coreai.llm.export", bundle_path(tmp_path, recipe))
    joined = " ".join(cmd)
    assert cmd[1] == "Qwen/Qwen3-0.6B"
    assert "--experimental" in joined
    assert "--compute-precision float16" in joined
    assert "--compression none" in joined
    # The fabric driver never gets --experimental (only coreai.llm.export needs it).
    fabric_cmd = build_command(recipe, "coreai-fabric-llm-export",
                               bundle_path(tmp_path, recipe))
    assert "--experimental" not in " ".join(fabric_cmd)


def test_coreai_llm_export_registry_name_uses_preset(tmp_path):
    # PRODUCTION path: apple_registry_name -> the short-name positional, and NO
    # --compute-precision/--compression/--experimental (the tested preset wins).
    recipe = _recipe(tool="coreai.llm.export")
    recipe.data["conversion"]["apple_registry_name"] = "qwen3-0.6b"
    cmd = build_command(recipe, "coreai.llm.export", bundle_path(tmp_path, recipe))
    joined = " ".join(cmd)
    assert cmd[1] == "qwen3-0.6b"  # registry short-name, not the HF id
    assert "--compute-precision" not in joined
    assert "--compression" not in joined
    assert "--experimental" not in joined
    assert "--output-name qwen3-0.6b" in joined
    assert "--overwrite" in joined


def test_known_tool_sets_are_consistent():
    assert REVISION_CAPABLE_TOOLS <= LLM_EXPORT_TOOLS


def test_script_tools_detected_and_explained():
    recipe = _recipe(tool="models/whisper/export.py",
                     args={"model": "openai/whisper-large-v3-turbo", "dtype": "float16"})
    assert is_script_tool("models/whisper/export.py")
    assert not is_script_tool("coreai-fabric-llm-export")
    hint = script_tool_hint(recipe, "models/whisper/export.py")
    assert "uv run" in hint
    assert "--model openai/whisper-large-v3-turbo" in hint
    assert "coreai-fabric verify qwen3-0.6b" in hint
