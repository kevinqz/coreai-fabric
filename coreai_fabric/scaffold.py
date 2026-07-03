"""`coreai-fabric new` — scaffold a draft recipe from upstream HF metadata."""
from __future__ import annotations

import re
from pathlib import Path

from . import hf
from .recipes import PERMISSIVE_LICENSES
from .util import err, find_root, ok, today, warn, write_yaml

#: HF pipeline_tag → (capabilities, input modalities, output modalities).
#: Deliberately small: only mappings we can state without guessing. Anything
#: else leaves the catalog block for the agent to fill before register.
PIPELINE_MAP = {
    "text-generation": (["text-generation"], ["text"], ["text"]),
    "automatic-speech-recognition": (["speech-to-text"], ["audio"], ["transcript"]),
    "depth-estimation": (["monocular-depth"], ["image"], ["depth_map"]),
    "image-classification": (["image-classification"], ["image"], ["labels"]),
    "object-detection": (["object-detection"], ["image"], ["boxes"]),
    "image-segmentation": (["segmentation"], ["image"], ["masks"]),
    "feature-extraction": (["embedding"], ["text"], ["embedding"]),
    "sentence-similarity": (["embedding"], ["text"], ["embedding"]),
    "text-to-speech": (["text-to-speech"], ["text"], ["audio"]),
}


def derive_id(hf_repo: str) -> str:
    base = hf_repo.split("/", 1)[1].lower()
    base = re.sub(r"[^a-z0-9.-]+", "-", base).strip("-")
    return base


def cmd_new(args) -> int:
    root = find_root()
    hf_repo = args.upstream_hf_repo
    if "/" not in hf_repo:
        err(f"'{hf_repo}' is not an owner/name HF repo id")
        return 1

    meta: dict = {}
    if args.offline:
        if not args.license:
            err("--offline requires --license (fabric never invents a license)")
            return 1
    else:
        try:
            meta = hf.model_info(hf_repo)
        except hf.HFError as exc:
            err(str(exc))
            print("hint: pass --offline with --license/--license-terms to scaffold without the network")
            return 1
        if meta.get("gated"):
            warn(f"{hf_repo} is gated on HF; conversion will require accepted terms + a token")

    recipe_id = args.id or derive_id(hf_repo)
    path = root / "recipes" / f"{recipe_id}.yaml"
    if path.exists() and not args.force:
        err(f"{path} already exists (use --force to overwrite)")
        return 1

    license_id = args.license or meta.get("license")
    if not license_id:
        err(f"{hf_repo} declares no license tag on HF; pass --license explicitly")
        return 1
    license_norm = license_id.strip().lower()
    if args.license_terms:
        license_terms = args.license_terms
    else:
        license_terms = "permissive" if license_norm in PERMISSIVE_LICENSES else "review_required"

    upstream: dict = {"hf_repo": hf_repo}
    if meta.get("sha"):
        upstream["revision"] = meta["sha"]
    upstream["license"] = license_norm
    upstream["license_terms"] = license_terms
    pipeline_tag = args.pipeline_tag or meta.get("pipeline_tag")
    if pipeline_tag:
        upstream["pipeline_tag"] = pipeline_tag
    if isinstance(meta.get("size_bytes"), int):
        upstream["size_bytes"] = meta["size_bytes"]

    data: dict = {
        "id": recipe_id,
        "upstream": upstream,
        "conversion": {
            "tool": args.tool,
            "args": {},
            "quantization": args.quantization,
            "precision": args.precision,
            "compute_units": args.compute_units,
        },
        "expected": {"bundle_files": args.bundle_file or ["metadata.json"]},
        "parity": {
            "gate_a": {
                "checks": [
                    "bundle_files_present",
                    "metadata_json_parses",
                    "metadata_matches_recipe",
                ]
            },
            "gate_b": _default_gate_b(pipeline_tag),
        },
        "publish": {
            "hf_target_namespace": args.namespace,
            "repo_name": args.repo_name or f"{recipe_id}-coreai",
        },
    }

    catalog_block = _catalog_block(args, hf_repo, pipeline_tag)
    if catalog_block:
        data["catalog"] = catalog_block
    else:
        warn(
            "no catalog block scaffolded (pipeline_tag "
            f"'{pipeline_tag}' has no honest mapping) — add one by hand before register"
        )

    data["status"] = "draft"
    if args.notes:
        data["notes"] = args.notes

    header = (
        f"# coreai-fabric recipe — scaffolded by `coreai-fabric new {hf_repo}` on {today()}\n"
        "# Status: draft. Review every field before convert; edit catalog: before register.\n"
    )
    write_yaml(path, data, header=header)
    ok(f"wrote {path.relative_to(root)}")
    print(f"next: coreai-fabric validate {recipe_id}")
    return 0


def _default_gate_b(pipeline_tag: str | None) -> dict:
    """Gate B defaults follow the zoo PORTING convention: cosine >= 0.999,
    plus greedy-token-exact for LLMs."""
    if pipeline_tag == "text-generation":
        return {
            "metric": "per_token_logit_cosine",
            "threshold": 0.999,
            "tolerance": 0.0005,
            "greedy_token_exact": True,
        }
    return {"metric": "graph_output_cosine", "threshold": 0.999, "tolerance": 0.0005}


def _catalog_block(args, hf_repo: str, pipeline_tag: str | None) -> dict | None:
    capabilities = list(args.capability or [])
    inputs = list(args.input or [])
    outputs = list(args.output or [])
    if not (capabilities and inputs and outputs):
        mapped = PIPELINE_MAP.get(pipeline_tag or "")
        if mapped:
            capabilities = capabilities or list(mapped[0])
            inputs = inputs or list(mapped[1])
            outputs = outputs or list(mapped[2])
    if not (capabilities and inputs and outputs):
        return None
    base_name = hf_repo.split("/", 1)[1]
    block: dict = {
        "name": args.name or f"{base_name} (fabric)",
        "family": args.family or base_name,
        "capabilities": capabilities,
        "modalities": {"input": inputs, "output": outputs},
    }
    # Everything below is unknowable before the first verified conversion.
    block["runner"] = "unknown"
    block["tokenizer_required"] = "unknown"
    block["processor_required"] = "unknown"
    block["aot_required"] = "unknown"
    return block
