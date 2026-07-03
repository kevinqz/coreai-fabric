"""The register generator must produce entries valid against the catalog
schemas (vendored snapshots in tests/fixtures/ — see its README) and honor
the shared fabric field contract."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from coreai_fabric.recipes import find_recipe
from coreai_fabric.register import (
    build_artifact_entry,
    build_model_entry,
    build_source_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"

FAKE_REVISION = "1f" * 20  # 40-hex, catalog schema requires a full sha
FAKE_FILES = [
    {"path": "qwen3-0.6b.aimodel/metadata.json", "sha256": "ab" * 32, "size_bytes": 512},
    {"path": "qwen3-0.6b.aimodel/weights.bin", "sha256": "cd" * 32, "size_bytes": 1200000000},
    {"path": "README.md", "sha256": "ef" * 32, "size_bytes": 2048},
]


def _published_recipe():
    recipe = find_recipe("qwen3-0.6b", REPO_ROOT)
    recipe = copy.deepcopy(recipe)
    recipe.data["status"] = "published"
    recipe.data["published"] = {
        "hf_repo": "coreai-community/qwen3-0.6b-coreai",
        "revision": FAKE_REVISION,
        "date": "2026-07-03",
    }
    return recipe


def _schema_errors(schema_file: str, entry: dict) -> list[str]:
    schema = json.loads((FIXTURES / schema_file).read_text())
    validator = Draft202012Validator(schema)
    return [
        f"{'.'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
        for e in validator.iter_errors(entry)
    ]


def test_model_entry_validates_against_catalog_schema():
    entry = build_model_entry(_published_recipe(), FAKE_FILES)
    assert _schema_errors("model.schema.json", entry) == []


def test_artifact_entry_validates_against_catalog_schema():
    entry = build_artifact_entry(_published_recipe(), FAKE_FILES, tool_version="0.0-test")
    assert _schema_errors("artifact.schema.json", entry) == []


def test_model_entry_shared_field_contract():
    entry = build_model_entry(_published_recipe(), FAKE_FILES)
    assert entry["source_group"] == "fabric"
    assert entry["artifact_ref"] == entry["id"] == "qwen3-0.6b"
    assert entry["source_path"].startswith("https://github.com/kevinqz/coreai-fabric/")
    # Facts fabric cannot know are unknown, never invented.
    assert entry["device_support"] == {
        "iphone": "unknown", "ipad": "unknown", "mac": "unknown", "mac_only": "unknown"
    }
    assert entry["license"] == {"name": "apache-2.0", "commercial_use": "likely"}
    assert entry["status"] == "needs_review"


def test_model_entry_satisfies_catalog_p1_invariants():
    # The catalog requires bundle_kind + min_os on every model and REJECTS
    # `unknown` for the four runtime facts (audit category 5). register must
    # emit all of them from the recipe, or the fabric->catalog lane can't complete.
    entry = build_model_entry(_published_recipe(), FAKE_FILES)
    assert entry["bundle_kind"] == "llm"
    assert entry["min_os"] == {"macos": "27.0", "ios": "27.0"}
    assert entry["upstream_repo"] == "Qwen/Qwen3-0.6B"
    rt = entry["runtime"]
    for field in ("stock_runtime", "custom_kernel", "patch_required", "aot_required"):
        assert isinstance(rt[field], bool), f"{field} must be a real bool, not {rt[field]!r}"


def test_artifact_entry_shared_field_contract():
    entry = build_artifact_entry(_published_recipe(), FAKE_FILES, tool_version="0.0-test")
    # HF-native provenance: no fabricated github block.
    assert "github" not in entry
    hf = entry["huggingface"]
    assert hf["owner"] == "coreai-community"
    assert hf["repo"] == "qwen3-0.6b-coreai"
    assert hf["revision"] == FAKE_REVISION
    assert hf["files"] == FAKE_FILES
    prov = entry["provenance"]
    assert prov["recipe_source"] == "fabric"
    # The verified converter executable (coreai-torch is a library, not a CLI).
    assert prov["converted_by"]["tool"] == "coreai-fabric-llm-export"
    assert prov["converted_by"]["version"] == "0.0-test"
    assert prov["converted_by"]["recipe_url"].endswith("/recipes/qwen3-0.6b.yaml")
    assert entry["officiality"] == {
        "apple_export_recipe": False,
        "apple_hosted_artifact": False,
        "community_packaged": True,
    }


def test_artifact_without_any_host_block_is_rejected_by_catalog_schema():
    entry = build_artifact_entry(_published_recipe(), FAKE_FILES, tool_version=None)
    del entry["huggingface"]
    errors = _schema_errors("artifact.schema.json", entry)
    assert errors, "anyOf(github, huggingface) must reject a host-less artifact"


def test_unknown_tool_version_stays_honest():
    entry = build_artifact_entry(_published_recipe(), FAKE_FILES, tool_version=None)
    assert entry["provenance"]["converted_by"]["version"] == "unknown"


def test_register_requires_published_block():
    recipe = find_recipe("qwen3-0.6b", REPO_ROOT)
    with pytest.raises(SystemExit, match="no published block"):
        build_artifact_entry(copy.deepcopy(recipe), FAKE_FILES, tool_version=None)


def test_review_required_license_maps_to_check_license():
    recipe = _published_recipe()
    recipe.data["upstream"]["license"] = "cc-by-nc-4.0"
    recipe.data["upstream"]["license_terms"] = "review_required"
    entry = build_model_entry(recipe, FAKE_FILES)
    assert entry["license"]["commercial_use"] == "check_license"
    assert _schema_errors("model.schema.json", entry) == []


def test_source_record_validates_against_catalog_source_schema():
    record = build_source_record()
    assert record["id"] == "coreai-fabric"
    assert _schema_errors("source.schema.json", record) == []
