"""The published model card's YAML frontmatter is the SotA discoverability
surface — it must be honest and correct, not just present."""
from __future__ import annotations

import copy
from pathlib import Path

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
    for t in ("coreai", "core-ai", "apple", "on-device", "llm", "4bit"):
        assert t in tags, f"missing tag {t}"
    assert len(tags) == len(set(tags)), "tags must be de-duplicated"


def test_card_omits_relation_for_uncompressed_export():
    # An uncompressed (`none`) export is NOT a quantized derivative — the card
    # must not claim base_model_relation: quantized rather than mislabel it.
    recipe = copy.deepcopy(find_recipe("qwen3-0.6b", REPO_ROOT))
    recipe.data["conversion"]["quantization"] = "none"
    fm = _frontmatter(recipe)
    assert "base_model_relation" not in fm
    assert "4bit" not in fm["tags"]


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
