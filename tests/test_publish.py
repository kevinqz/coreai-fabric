"""The published model card's YAML frontmatter is the SotA discoverability
surface — it must be honest and correct, not just present."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from coreai_fabric.publish import render_model_card
from coreai_fabric.recipes import find_recipe

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = {"tool": "coreai.llm.export", "tool_version": None, "input": {"revision": "abc123"}}
REPORT = {"gate_a": {"status": "passed"},
          "gate_b": {"metric": "benchmark_accuracy", "threshold": 0.999,
                     "status": "not_run", "value": None}}


def _frontmatter(recipe) -> dict:
    card = render_model_card(REPO_ROOT, recipe, MANIFEST, REPORT)
    assert card.startswith("---\n")
    fm = card.split("---\n", 2)[1]
    return yaml.safe_load(fm)


def test_card_frontmatter_is_sota_for_quantized_asset():
    fm = _frontmatter(find_recipe("qwen3-0.6b", REPO_ROOT))
    assert fm["base_model"] == "Qwen/Qwen3-0.6B"
    # The correct relation for a quantized export — NOT `finetune` (HF's default
    # and what the coreai-community cards wrongly show).
    assert fm["base_model_relation"] == "quantized"
    assert fm["library_name"] == "coreai"
    tags = fm["tags"]
    # Accurate descriptors that are also the ecosystem's discoverability facets.
    for t in ("coreai", "core-ai", "coreml", "apple", "apple-silicon",
              "on-device", "iphone", "metal", "text-generation", "llm", "4bit"):
        assert t in tags, f"missing tag {t}"
    assert len(tags) == len(set(tags)), "tags must be de-duplicated"
    # Curated example prompts render as widget examples (real usage docs).
    assert fm.get("widget") and all("text" in w for w in fm["widget"])


def test_card_omits_relation_for_uncompressed_export():
    # An uncompressed (`none`) export is NOT a quantized derivative — the card
    # must not claim base_model_relation: quantized rather than mislabel it.
    recipe = copy.deepcopy(find_recipe("qwen3-0.6b", REPO_ROOT))
    recipe.data["conversion"]["quantization"] = "none"
    fm = _frontmatter(recipe)
    assert "base_model_relation" not in fm
    assert "4bit" not in fm["tags"]


def test_card_never_advertises_an_unpublished_sibling_variant():
    # Publishing int8 while its int4 sibling is only drafted (no `published`
    # block) must NOT put an `int4/` row on the card — that would advertise a
    # repo subdir that 404s. A lone real tier drops the comparison table.
    int8 = find_recipe("qwen3-0.6b-int8", REPO_ROOT)
    card = render_model_card(REPO_ROOT, int8, MANIFEST, REPORT)
    assert "## Quantization variants" not in card
    assert "`int4/`" not in card


def test_mirror_dry_run_plans_source_to_target(capsys):
    # S3: mirror maps the published canonical repo to the org copy, source of
    # truth preserved. Dry-run copies nothing.
    from types import SimpleNamespace

    from coreai_fabric.publish import cmd_mirror
    rc = cmd_mirror(SimpleNamespace(id="qwen3-0.6b-int8", to="coreai-community", dry_run=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "kevinqz/Qwen3-0.6B-CoreAI" in out
    assert "coreai-community/Qwen3-0.6B-CoreAI" in out


def test_mirror_refuses_a_recipe_without_published_block():
    # You can only mirror what's already published to your namespace.
    from types import SimpleNamespace

    from coreai_fabric.publish import cmd_mirror
    rc = cmd_mirror(SimpleNamespace(id="qwen3-4b", to="coreai-community", dry_run=True))
    assert rc == 1


def test_publish_and_register_flag_parses():
    # S1: publish carries the seamless-flow flags (chaining is exercised e2e, not
    # here — it opens a real PR — but the flag must exist so the path is reachable).
    from coreai_fabric.cli import build_parser
    args = build_parser().parse_args(
        ["publish", "qwen3-0.6b-int8", "--and-register", "--catalog-path", "/tmp/cat"])
    assert args.and_register is True
    assert args.catalog_path == "/tmp/cat"


def test_card_refuses_to_mislabel_a_non_llm_bundle():
    # S2: the LLM card template (chat hook, KV-cache prose, CoreAILanguageModel
    # example) must NOT render for a non-LLM bundle — a whisper .aimodel is not a
    # chat model. fabric fails loud with the fix instead of shipping a lying card.
    recipe = find_recipe("whisper-large-v3-turbo", REPO_ROOT)  # bundle_kind: asr
    with pytest.raises(SystemExit, match="bundle_kind 'asr'"):
        render_model_card(REPO_ROOT, recipe, MANIFEST, REPORT)


class _FakeCollection:
    slug = "kevinqz/coreai-apple-on-device-abc123"


class _FakeApi:
    def __init__(self):
        self.created = None
        self.added = None

    def create_collection(self, title, *, namespace, exists_ok):
        self.created = (title, namespace, exists_ok)
        return _FakeCollection()

    def add_collection_item(self, slug, *, item_id, item_type, exists_ok):
        self.added = (slug, item_id, item_type, exists_ok)


def test_add_to_collection_creates_and_adds_idempotently():
    from coreai_fabric.publish import _add_to_collection
    api = _FakeApi()
    url = _add_to_collection(api, "kevinqz", "CoreAI · Apple on-device",
                             "kevinqz/Qwen3-0.6B-CoreAI")
    assert api.created == ("CoreAI · Apple on-device", "kevinqz", True)
    assert api.added == ("kevinqz/coreai-apple-on-device-abc123",
                         "kevinqz/Qwen3-0.6B-CoreAI", "model", True)
    assert url == "https://huggingface.co/collections/kevinqz/coreai-apple-on-device-abc123"


def test_add_to_collection_never_fails_a_completed_publish():
    # The model is already uploaded when this runs — a Collections error must
    # warn and return None, not raise.
    from coreai_fabric.publish import _add_to_collection

    class _Boom:
        def create_collection(self, *a, **k):
            raise RuntimeError("403 not a member")

    assert _add_to_collection(_Boom(), "kevinqz", "CoreAI", "kevinqz/X-CoreAI") is None


def test_sanitize_manifest_strips_host_paths():
    # The public manifest must not leak the OS username / local dir layout.
    from coreai_fabric.publish import sanitize_manifest
    root = Path("/Users/someone/Dev/coreai-fabric")
    manifest = {
        "tool": "coreai.llm.export",
        "tool_path": "/Users/someone/Dev/coreai-fabric/.venv/bin/coreai.llm.export",
        "command": ["/Users/someone/Dev/coreai-fabric/.venv/bin/coreai.llm.export",
                    "qwen3-0.6b", "--output-dir",
                    "/Users/someone/Dev/coreai-fabric/build"],
        "input": {"hf_repo": "Qwen/Qwen3-0.6B"},
    }
    out = sanitize_manifest(manifest, root)
    assert "tool_path" not in out
    assert out["command"][0] == "coreai.llm.export"          # basename only
    assert "/Users/someone" not in json.dumps(out)           # no absolute local path
    assert "./build" in out["command"]                       # root relativized to '.'


def test_assert_no_local_paths_catches_leak(tmp_path):
    from coreai_fabric.publish import assert_no_local_paths
    (tmp_path / "clean.json").write_text('{"a": 1}')
    assert assert_no_local_paths(tmp_path) == []
    (tmp_path / "leak.json").write_text('{"p": "/Users/kevin/x"}')
    assert assert_no_local_paths(tmp_path) == ["leak.json"]


def test_copyright_holder_parsed_from_license(tmp_path):
    from coreai_fabric.publish import copyright_holder_from_license
    (tmp_path / "LICENSE").write_text("Apache License 2.0\n\n   Copyright 2024 Alibaba Cloud\n")
    assert copyright_holder_from_license(tmp_path) == "2024 Alibaba Cloud"
    # A bare template placeholder must not be mistaken for a real holder.
    (tmp_path / "LICENSE").write_text("Copyright [yyyy] [name of copyright owner]\n")
    assert copyright_holder_from_license(tmp_path) is None
