"""Recipe loading, schema validation (aggregated errors), and license triage."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from jsonschema import Draft202012Validator

from .util import find_root, read_yaml

STATUSES = ["draft", "converted", "verified", "published", "registered"]

#: Licenses fabric treats as permissive for triage purposes. Deliberately small
#: and conservative; anything else is review_required. Triage labels are not
#: legal advice (same stance as coreai-catalog's commercial_use field).
PERMISSIVE_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "cc0-1.0",
    "unlicense",
}


@dataclass
class Issue:
    """One validation finding, addressed to an agent: path + message + hint."""

    recipe: str
    severity: str  # "error" | "warning"
    path: str
    message: str
    hint: str | None = None

    def render(self) -> str:
        loc = f"{self.recipe}:{self.path}" if self.path else self.recipe
        line = f"[{self.severity}] {loc}: {self.message}"
        if self.hint:
            line += f"\n    hint: {self.hint}"
        return line


@dataclass
class Recipe:
    path: Path
    data: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.data.get("id", self.path.stem)

    @property
    def status(self) -> str:
        return self.data.get("status", "draft")


def recipe_schema(root: Path | None = None) -> dict:
    root = root or find_root()
    return json.loads((root / "schema" / "recipe.schema.json").read_text())


def load_recipe(path: Path) -> Recipe:
    return Recipe(path=path, data=read_yaml(path))


def load_all_recipes(root: Path | None = None) -> list[Recipe]:
    root = root or find_root()
    recipes_dir = root / "recipes"
    return [load_recipe(p) for p in sorted(recipes_dir.glob("*.yaml"))]


def find_recipe(recipe_id: str, root: Path | None = None) -> Recipe:
    root = root or find_root()
    path = root / "recipes" / f"{recipe_id}.yaml"
    if not path.is_file():
        available = [p.stem for p in sorted((root / "recipes").glob("*.yaml"))]
        raise SystemExit(
            f"error: no recipe '{recipe_id}' (expected {path}).\n"
            f"Available recipes: {', '.join(available) or '<none>'}"
        )
    return load_recipe(path)


def validate_recipe(recipe: Recipe, schema: dict) -> list[Issue]:
    """Schema + consistency validation. Returns ALL issues, never fail-fast."""
    issues: list[Issue] = []
    name = recipe.path.name
    validator = Draft202012Validator(schema)
    for error in sorted(validator.iter_errors(recipe.data), key=lambda e: list(e.absolute_path)):
        path = ".".join(str(p) for p in error.absolute_path)
        hint = None
        if error.validator == "enum":
            hint = f"allowed values: {', '.join(repr(v) for v in error.validator_value)}"
        elif error.validator == "required":
            hint = "add the missing field; see schema/recipe.schema.json for its shape"
        elif error.validator == "additionalProperties":
            hint = "remove the unrecognized field (the recipe contract is strict)"
        issues.append(Issue(name, "error", path, error.message, hint))

    data = recipe.data
    if isinstance(data.get("id"), str) and data["id"] != recipe.path.stem:
        issues.append(
            Issue(
                name,
                "error",
                "id",
                f"id '{data['id']}' does not match filename stem '{recipe.path.stem}'",
                hint=f"rename the file to recipes/{data['id']}.yaml or fix the id",
            )
        )

    status = data.get("status")
    has_published = "published" in data
    if status in ("published", "registered") and not has_published:
        issues.append(
            Issue(
                name,
                "error",
                "published",
                f"status '{status}' requires a published block (hf_repo, revision, date)",
                hint="`coreai-fabric publish` writes this block; do not set the status by hand",
            )
        )
    if has_published and status in ("draft", "converted", "verified"):
        issues.append(
            Issue(
                name,
                "error",
                "status",
                f"a published block is present but status is '{status}'",
                hint="published recipes must have status published or registered",
            )
        )

    issues.extend(triage_license(recipe))
    return issues


def triage_license(recipe: Recipe) -> list[Issue]:
    """License triage. review_required upstreams are flagged (warning), and a
    permissive claim for a license outside the allowlist is an error."""
    issues: list[Issue] = []
    name = recipe.path.name
    upstream = recipe.data.get("upstream")
    if not isinstance(upstream, dict):
        return issues
    license_id = upstream.get("license")
    terms = upstream.get("license_terms")
    normalized = license_id.strip().lower() if isinstance(license_id, str) else ""

    if terms == "permissive" and normalized not in PERMISSIVE_LICENSES:
        issues.append(
            Issue(
                name,
                "error",
                "upstream.license_terms",
                f"license '{license_id}' is not on the fabric permissive allowlist "
                f"but license_terms claims 'permissive'",
                hint=(
                    "set license_terms: review_required (allowlist: "
                    + ", ".join(sorted(PERMISSIVE_LICENSES))
                    + ")"
                ),
            )
        )
    if terms == "review_required":
        issues.append(
            Issue(
                name,
                "warning",
                "upstream.license_terms",
                f"upstream license '{license_id}' requires review before publish",
                hint="publish will refuse without --acknowledge-license-review",
            )
        )
    if terms == "unknown":
        issues.append(
            Issue(
                name,
                "warning",
                "upstream.license_terms",
                "license terms not yet triaged",
                hint="resolve the upstream license and set permissive or review_required",
            )
        )
    if terms == "restricted":
        # A restricted upstream must NEVER have its converted weights republished.
        # It previously had no branch at all — a recipe hand-set to `restricted`
        # sailed past triage AND publish (a silent weights redistribution). Flag
        # it, and publish hard-refuses the weights path even with an ack.
        issues.append(
            Issue(
                name,
                "warning",
                "upstream.license_terms",
                f"upstream license '{license_id}' is restricted — its weights may NOT be republished",
                hint="publish refuses the weights path for restricted (no --acknowledge bypass); "
                "index the upstream + ship the recipe instead",
            )
        )
    if normalized in PERMISSIVE_LICENSES and terms == "review_required":
        issues.append(
            Issue(
                name,
                "warning",
                "upstream.license_terms",
                f"license '{license_id}' is on the permissive allowlist but marked review_required",
                hint="this is allowed (stricter is fine) — drop this note by setting permissive",
            )
        )
    return issues


def commercial_use_for(recipe: Recipe) -> str:
    """Map fabric license triage to the catalog's commercial_use vocabulary."""
    terms = (recipe.data.get("upstream") or {}).get("license_terms")
    return "likely" if terms == "permissive" else "check_license"
