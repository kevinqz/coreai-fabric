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


def test_model_entry_carries_measured_parity_into_catalog():
    # C1/E3/C2: the measured greedy_parity signature reaches the catalog as data
    # (not a note), with a fidelity_tier and a variant_group — and still validates.
    recipe = _published_recipe()
    recipe.data["publish"]["variant"] = "int8"
    report = {
        "gate_a": {"status": "passed"},
        "gate_b": {"metric": "greedy_parity", "status": "passed", "value": 1.0,
                   "margin_gated_match_rate": 1.0, "margin_gated_ci95": [0.9, 1.0],
                   "argmax_match_rate": 0.958, "top5_agreement_rate": 1.0,
                   "matched": 46, "compared": 48, "greedy_token_exact": False,
                   "reference_dtype": "float16", "flip_margin_nats": 0.1,
                   "runner": "coreai-fabric-parity-runner/0.1.0",
                   "environment": {"chip": "Apple M4 Max", "os": "macOS 26.6"}},
    }
    entry = build_model_entry(recipe, FAKE_FILES, report=report)
    assert _schema_errors("model.schema.json", entry) == []
    assert entry["evaluation"]["metric"] == "greedy_parity"
    assert entry["evaluation"]["argmax_match_rate"] == 0.958
    assert entry["evaluation"]["measured_on"].startswith("Apple M4 Max")
    assert entry["size"]["fidelity_tier"] == "high_fidelity"   # Gate B passed
    assert entry["variant_group"] == "kevinqz/Qwen3-0.6B-CoreAI"
    # A failed report → the size tier, honestly.
    report["gate_b"]["status"] = "failed"
    assert build_model_entry(recipe, FAKE_FILES, report=report)["size"]["fidelity_tier"] == "size"


def test_llm_model_entry_carries_a_typed_io_contract():
    # C4: a fabric LLM is as agent-ready as the official entries — a typed
    # io_contract (CoreAILanguageModel entrypoint, text->text, stateful+streaming
    # session, tokenizer ref), derived truthfully from the recipe.
    recipe = _published_recipe()
    entry = build_model_entry(recipe, FAKE_FILES)
    io = entry["io_contract"]
    assert io["entrypoint"]["type"] == "CoreAILanguageModel"
    assert io["inputs"][0]["modality"] == "text"
    assert io["outputs"][0]["swift_type"] == "String"
    assert io["session"] == {"stateful": True, "streaming": True}
    ctx = recipe.data["catalog"].get("context_length")
    if ctx:
        assert io["inputs"][0]["constraints"]["max_context"] == ctx
    if recipe.data["catalog"].get("tokenizer_required"):
        assert io["files"]["tokenizer_ref"] == "macos/tokenizer"
    # The upstream repo is named in the detokenization note (no invented vocab).
    assert recipe.data["upstream"]["hf_repo"] in io["outputs"][0]["decoding"]["detokenization"]
    assert _schema_errors("model.schema.json", entry) == []


def test_action_model_carries_a_typed_io_contract():
    # VLA lane: a robot policy is agent-ready too — obs modalities in, an action
    # chunk out, NON-stateful (not chat). Emitted so the catalog's E6 test accepts
    # a fabric VLA instead of rejecting it as untyped.
    recipe = _published_recipe()
    recipe.data["catalog"]["bundle_kind"] = "action"
    recipe.data["catalog"]["capabilities"] = ["vision-language-action", "robotics"]
    recipe.data["catalog"]["modalities"] = {"input": ["image", "text", "state"], "output": ["action"]}
    recipe.data["catalog"]["processor_required"] = True
    entry = build_model_entry(recipe, FAKE_FILES)
    io = entry["io_contract"]
    assert io["entrypoint"]["type"] == "CoreAIRunner"
    assert io["session"] == {"stateful": False, "streaming": False}   # NOT chat
    assert io["outputs"][0]["name"] == "action_chunk"
    assert {i["modality"] for i in io["inputs"]} == {"image", "text", "state"}
    assert io["files"]["processor_ref"] == "norm_stats.json"
    assert _schema_errors("model.schema.json", entry) == []


def test_non_llm_bundle_gets_no_fabricated_io_contract():
    # C4 is honest: a bundle kind fabric can't yet describe truthfully gets NO
    # io_contract rather than a wrong one — the catalog's fabric-io_contract
    # test then forces a real one to be authored before it can be registered.
    recipe = _published_recipe()
    recipe.data["catalog"]["bundle_kind"] = "asr"
    entry = build_model_entry(recipe, FAKE_FILES)
    assert "io_contract" not in entry


def test_recipe_traits_reach_catalog_as_separate_facet():
    # Architecture/inference traits (moe, mla, …) are emitted as a `traits`
    # facet, never mixed into the capabilities vocabulary — and still validate.
    recipe = _published_recipe()
    recipe.data["catalog"]["traits"] = ["moe"]
    entry = build_model_entry(recipe, FAKE_FILES)
    assert _schema_errors("model.schema.json", entry) == []
    assert entry["traits"] == ["moe"]
    assert "moe" not in entry["capabilities"]   # a trait never leaks into tasks


def test_recipe_without_traits_omits_the_facet():
    # Most models have no special trait; register must not emit an empty list.
    entry = build_model_entry(_published_recipe(), FAKE_FILES)
    assert "traits" not in entry


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
    # The recipe's real converter — the seed uses Apple's PRODUCTION CLI
    # (coreai.llm.export), verified on hardware to emit the KV-cache chat asset.
    # register records whatever the recipe declares; it never invents a tool.
    assert prov["converted_by"]["tool"] == "coreai.llm.export"
    assert prov["converted_by"]["version"] == "0.0-test"
    assert prov["converted_by"]["recipe_url"].endswith("/recipes/qwen3-0.6b.yaml")
    assert entry["officiality"] == {
        "apple_export_recipe": False,
        "apple_hosted_artifact": False,
        "community_packaged": True,
    }


def test_variant_artifact_url_follows_catalog_subdir_convention():
    # A variant tier lives in a `<variant>/` subdir of the shared repo. The
    # artifact's hf.url + path must match the catalog's existing convention
    # (gemma-4-e2b-vision, efficientsam3): path == "tree/main/<variant>" and
    # url == base + "/" + path — otherwise the catalog's URL-consistency audit
    # (`url == base/path`) flags it, as it did on the first int8 publish.
    recipe = _published_recipe()
    recipe.data["publish"]["variant"] = "int8"
    entry = build_artifact_entry(recipe, FAKE_FILES, tool_version="0.0-test")
    hf = entry["huggingface"]
    assert hf["path"] == "tree/main/int8"
    assert hf["url"] == "https://huggingface.co/coreai-community/qwen3-0.6b-coreai/tree/main/int8"
    assert _schema_errors("artifact.schema.json", entry) == []


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
