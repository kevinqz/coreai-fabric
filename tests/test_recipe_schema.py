"""Recipe contract tests: seed recipes green, garbage rejected with
aggregated errors, license triage behaves."""
from __future__ import annotations

from pathlib import Path

from coreai_fabric.recipes import (
    Recipe,
    load_all_recipes,
    recipe_schema,
    triage_license,
    validate_recipe,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = recipe_schema(REPO_ROOT)


def test_seed_recipes_exist():
    recipes = load_all_recipes(REPO_ROOT)
    ids = {r.id for r in recipes}
    assert {"qwen3-0.6b", "whisper-large-v3-turbo", "da3-small"} <= ids


def test_all_seed_recipes_validate_without_errors():
    for recipe in load_all_recipes(REPO_ROOT):
        issues = validate_recipe(recipe, SCHEMA)
        errors = [i for i in issues if i.severity == "error"]
        assert not errors, f"{recipe.id}: " + "; ".join(i.render() for i in errors)


def test_seed_recipes_are_honest_drafts():
    # Committed seed recipes stay `draft`: converted/verified describe LOCAL
    # build/ state (disposable, gitignored), so they are reset before commit.
    # Real conversion/verification runs are recorded in docs/validation-log.md,
    # not in the status field. Status leaves draft in the committed tree ONLY
    # at publish — and then an honest `published` block (hf_repo + revision)
    # must back the claim with a durable artifact.
    PUBLISHED_STATES = {"published", "registered"}
    for recipe in load_all_recipes(REPO_ROOT):
        pub = recipe.data.get("published")
        if recipe.status in PUBLISHED_STATES:
            assert pub and pub.get("hf_repo") and pub.get("revision"), (
                f"{recipe.id} is '{recipe.status}' but has no published block "
                "(hf_repo + revision) — a published status must point to a "
                "durable artifact, never an empty claim"
            )
        else:
            assert recipe.status == "draft", (
                f"{recipe.id} is committed with status '{recipe.status}' — "
                "converted/verified refer to disposable build/ state; reset to "
                "draft before committing (see docs/validation-log.md)"
            )
            assert pub is None, f"{recipe.id} is draft but carries a published block"


def test_garbage_recipe_rejected_with_aggregated_errors():
    garbage = Recipe(
        path=Path("recipes/garbage.yaml"),
        data={
            "id": "Not A Valid Id",
            "upstream": {"hf_repo": "no-slash", "license_terms": "vibes"},
            "conversion": {"tool": ""},
            "status": "cooked",
            "surprise_field": True,
        },
    )
    issues = validate_recipe(garbage, SCHEMA)
    errors = [i for i in issues if i.severity == "error"]
    # Aggregation: multiple independent defects must surface in ONE pass.
    assert len(errors) >= 5, [i.render() for i in errors]
    messages = " | ".join(i.message for i in errors)
    assert "surprise_field" in messages  # additionalProperties
    assert "cooked" in messages  # status enum
    assert any(i.hint for i in errors)  # errors carry fix hints


def test_id_filename_mismatch_is_an_error():
    recipe = Recipe(path=Path("recipes/other-name.yaml"), data=_minimal_valid(id="real-id"))
    issues = validate_recipe(recipe, SCHEMA)
    assert any("does not match filename" in i.message for i in issues if i.severity == "error")


def test_published_status_requires_published_block():
    data = _minimal_valid(id="x")
    data["status"] = "published"
    issues = validate_recipe(Recipe(path=Path("recipes/x.yaml"), data=data), SCHEMA)
    assert any("requires a published block" in i.message for i in issues if i.severity == "error")


def test_license_triage_rejects_permissive_overclaim():
    data = _minimal_valid(id="x")
    data["upstream"]["license"] = "cc-by-nc-4.0"
    data["upstream"]["license_terms"] = "permissive"
    issues = triage_license(Recipe(path=Path("recipes/x.yaml"), data=data))
    assert any(i.severity == "error" for i in issues)


def test_license_triage_flags_review_required_as_warning_not_error():
    data = _minimal_valid(id="x")
    data["upstream"]["license"] = "cc-by-nc-4.0"
    data["upstream"]["license_terms"] = "review_required"
    issues = triage_license(Recipe(path=Path("recipes/x.yaml"), data=data))
    assert any(i.severity == "warning" for i in issues)
    assert not any(i.severity == "error" for i in issues)


def _minimal_valid(id: str) -> dict:
    return {
        "id": id,
        "upstream": {
            "hf_repo": "some-org/some-model",
            "license": "apache-2.0",
            "license_terms": "permissive",
        },
        "conversion": {
            "tool": "coreai-fabric-llm-export",
            "quantization": "none",
            "precision": "float16",
        },
        "expected": {"bundle_files": ["metadata.json"]},
        "parity": {
            "gate_a": {"checks": ["bundle_files_present"]},
            "gate_b": {"metric": "graph_output_cosine", "threshold": 0.999, "tolerance": 0.0005},
        },
        "publish": {"hf_target_namespace": "some-org", "repo_name": "some-model-coreai"},
        "status": "draft",
    }
