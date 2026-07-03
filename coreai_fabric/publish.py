"""`coreai-fabric publish` — upload the build output to the publisher's own
HF namespace and record the result back into the recipe.

Fabric never hosts weights: the target is always the publisher's namespace
(publish.hf_target_namespace in the recipe). Requires huggingface_hub
(pip install "coreai-fabric[hf]") and a logged-in token (`hf auth login`).
"""
from __future__ import annotations

import json
from pathlib import Path

from .convert import bundle_path, manifest_path
from .recipes import find_recipe, triage_license
from .util import err, find_root, ok, today, warn, write_yaml
from .verify import parity_report_path


def render_model_card(root: Path, recipe, manifest: dict, report: dict) -> str:
    template = (root / "templates" / "model-card.md").read_text()
    upstream = recipe.data["upstream"]
    conv = recipe.data["conversion"]
    gate_b = report["gate_b"]
    from . import FABRIC_REPO_URL

    catalog_block = recipe.data.get("catalog") or {}

    # SotA HF discoverability metadata. base_model_relation: a CoreAI export is
    # a `quantized` derivative when the preset compresses weights — NOT a
    # `finetune` (HF's default, and what the community's own cards wrongly show).
    # Omit the relation for an uncompressed (`none`) export rather than mislabel.
    quant = str(conv.get("quantization", "none")).strip()
    is_quantized = quant.lower() not in ("", "none")
    base_model_relation_line = (
        "base_model_relation: quantized\n" if is_quantized else ""
    )
    # Consistent, de-duped tag set (aligns with coreai-community's vocabulary:
    # core-ai / coreai / on-device / apple-silicon) plus honest bundle_kind and
    # quantization so the asset is findable by exactly what it is.
    tags = ["coreai", "core-ai", "coreai-fabric", "aimodel",
            "apple", "apple-silicon", "on-device"]
    if catalog_block.get("bundle_kind"):
        tags.append(str(catalog_block["bundle_kind"]))
    if is_quantized:
        tags.append(quant.lower())
    seen: set[str] = set()
    tags_block = "".join(
        f"- {t}\n" for t in tags if not (t in seen or seen.add(t))
    )

    return template.format(
        license=upstream["license"],
        upstream_hf_repo=upstream["hf_repo"],
        upstream_revision=manifest.get("input", {}).get("revision")
        or upstream.get("revision", "unpinned"),
        pipeline_tag=upstream.get("pipeline_tag", ""),
        base_model_relation_line=base_model_relation_line,
        tags_block=tags_block,
        name=catalog_block.get("name", recipe.id),
        recipe_id=recipe.id,
        recipe_url=f"{FABRIC_REPO_URL}/blob/main/recipes/{recipe.id}.yaml",
        tool=manifest.get("tool", conv["tool"]),
        tool_version=manifest.get("tool_version") or "(version not reported)",
        precision=conv["precision"],
        quantization=conv["quantization"],
        date=today(),
        gate_a_status=report["gate_a"]["status"],
        gate_b_metric=gate_b["metric"],
        gate_b_threshold=gate_b["threshold"],
        gate_b_status=gate_b["status"],
        gate_b_value=f" (value: {gate_b['value']})" if gate_b.get("value") is not None else "",
    )


def _add_to_collection(api, namespace: str, title: str, repo_id: str) -> str | None:
    """Ensure a namespace Collection exists and add the published repo to it.

    Best-effort by design: the model is already uploaded when this runs, so a
    Collections hiccup (permissions, API) warns and returns None rather than
    failing a completed publish. Idempotent via exists_ok on both calls."""
    try:
        coll = api.create_collection(title=title, namespace=namespace, exists_ok=True)
        api.add_collection_item(coll.slug, item_id=repo_id, item_type="model", exists_ok=True)
        return f"https://huggingface.co/collections/{coll.slug}"
    except Exception as exc:  # noqa: BLE001 — a completed upload must not fail on this
        warn(f"published, but could not add to collection {title!r}: {exc}")
        return None


def cmd_publish(args) -> int:
    root = find_root()
    recipe = find_recipe(args.id, root)

    # Preconditions: verified build output + clean license triage.
    bundle = bundle_path(root, recipe)
    if not bundle.is_dir():
        err(f"no bundle at {bundle.relative_to(root)} — run convert + verify first")
        return 1
    rpath = parity_report_path(root, recipe)
    if not rpath.is_file():
        err(f"no parity report at {rpath.relative_to(root)} — run `coreai-fabric verify {recipe.id}` first")
        return 1
    report = json.loads(rpath.read_text())
    if report["gate_a"]["status"] != "passed":
        err("Gate A has not passed — refusing to publish an unverified bundle")
        return 1
    if report["gate_b"]["status"] != "passed":
        if args.allow_unverified_parity:
            warn(
                "publishing WITHOUT numeric parity (gate B "
                f"{report['gate_b']['status']}) — the model card will say so"
            )
        else:
            err(
                "Gate B has not passed. Numeric parity is the point of fabric; "
                "pass --allow-unverified-parity only if you accept publishing "
                "a structurally-valid but numerically-unproven artifact."
            )
            return 1

    license_errors = [i for i in triage_license(recipe) if i.severity == "error"]
    if license_errors:
        for issue in license_errors:
            err(issue.render())
        return 1
    if recipe.data["upstream"]["license_terms"] == "review_required" and not args.acknowledge_license_review:
        err(
            "upstream license is review_required — a human must review the "
            "license terms, then re-run with --acknowledge-license-review"
        )
        return 1

    mpath = manifest_path(root, recipe)
    manifest = json.loads(mpath.read_text()) if mpath.is_file() else {}
    card = render_model_card(root, recipe, manifest, report)

    publish_cfg = recipe.data["publish"]
    repo_id = f"{publish_cfg['hf_target_namespace']}/{publish_cfg['repo_name']}"

    if args.dry_run:
        print(f"would publish to https://huggingface.co/{repo_id}")
        print(f"  bundle: {bundle.relative_to(root)}")
        print(f"  extras: README.md (model card), parity-report.json, conversion-manifest.json")
        print("--- model card ---")
        print(card)
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        err(
            "huggingface_hub is not installed. Install the publish extra:\n"
            '  pip install "coreai-fabric[hf]"\n'
            "then authenticate with `hf auth login`."
        )
        return 1

    api = HfApi()
    staging = root / "build" / recipe.id / "publish-staging"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "README.md").write_text(card)
    (staging / "parity-report.json").write_text(json.dumps(report, indent=2) + "\n")
    if manifest:
        (staging / "conversion-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"creating repo (if missing): {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"uploading bundle {bundle.name} ...")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(bundle),
        path_in_repo=bundle.name,
        commit_message=f"coreai-fabric publish {recipe.id}",
    )
    info = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(staging),
        path_in_repo=".",
        commit_message=f"coreai-fabric publish {recipe.id}: card + reports",
    )
    revision = getattr(info, "oid", None) or api.model_info(repo_id).sha

    # Organize within the namespace: drop the model into a Collection so a
    # publisher's CoreAI work is grouped and separated from the rest of their
    # HF repos (HF namespaces are flat; Collections are the native grouping).
    collection_title = publish_cfg.get("collection")
    if collection_title:
        url = _add_to_collection(
            api, publish_cfg["hf_target_namespace"], collection_title, repo_id
        )
        if url:
            ok(f"added to collection '{collection_title}': {url}")

    recipe.data["published"] = {
        "hf_repo": repo_id,
        "revision": revision,
        "date": today(),
    }
    recipe.data["status"] = "published"
    write_yaml(recipe.path, recipe.data)
    ok(f"published https://huggingface.co/{repo_id} @ {revision}")
    ok(f"recipe status -> published ({recipe.path.name})")
    print(f"next: coreai-fabric register {recipe.id} --catalog-path ../coreai-catalog --dry-run")
    return 0
