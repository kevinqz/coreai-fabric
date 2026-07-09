"""`coreai-fabric register` — generate coreai-catalog entries for a published
artifact and open a PR against github.com/kevinqz/coreai-catalog.

Implements the shared field contract between fabric and the catalog:
  - model entry: source_group: fabric
  - artifact entry: NO github block (huggingface-only provenance, allowed by
    the catalog's anyOf(github, huggingface)), huggingface.revision +
    huggingface.files digests, and a provenance block with
    converted_by {tool, version, recipe_url} + recipe_source: fabric.

Entries are validated against the catalog schemas read from --catalog-path,
so register fails loudly (not silently) if the target catalog clone predates
the contract.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

from . import FABRIC_REPO_URL
from .convert import manifest_path
from .recipes import Recipe, commercial_use_for, find_recipe
from .util import dump_yaml, err, find_root, ok, today, warn, write_yaml
from .verify import parity_report_path

SOURCE_RECORD_ID = "coreai-fabric"


def _bool_or_unknown(value) -> object:
    return value if value in (True, False) else "unknown"


def _humanize_bytes(total: int) -> str:
    if total >= 1 << 30:
        return f"{total / (1 << 30):.1f}GB"
    if total >= 1 << 20:
        return f"{total / (1 << 20):.1f}MB"
    return f"{total}B"


def recipe_url(recipe: Recipe) -> str:
    return f"{FABRIC_REPO_URL}/blob/main/recipes/{recipe.id}.yaml"


#: Fields of a gate_b report that the LIVE coreai-catalog `evaluation` schema
#: accepts today (verified against ../coreai-catalog 2026-07-09). These carry the
#: measured parity numbers to the catalog without breaking the cross-contract
#: check. The catalog schema's `additionalProperties: false` means anything not
#: in this set is REJECTED — so we emit only the accepted vocabulary.
_EVAL_KEYS_ACCEPTED = (
    "metric", "status", "reason",
    # greedy_parity family
    "margin_gated_match_rate", "margin_gated_ci95", "argmax_match_rate",
    "top5_agreement_rate", "matched", "compared", "greedy_token_exact",
    "reference_dtype", "flip_margin_nats",
    # action_parity family (catalog-accepted names)
    "n_obs", "min_chunk_cosine", "mean_chunk_cosine",
    "max_action_mae", "max_relative_action_mae",
    # framework provenance
    "framework", "framework_version",
    "runner", "measured_on",
)

#: The Phase-0 protocol signature + generic numeric core (RFC F2/F6) that the
#: catalog schema does NOT yet accept. These are the documented batched follow-up
#: PR to coreai-catalog (RFC §4.3). We keep them in the RECIPE (the durable
#: fabric-side home) and the generated docs/scorecard.md; they reach the catalog
#: only after the schema PR lands. `catalog_protocol_extension()` returns them so
#: the follow-up PR is a one-call wire-up.
_EVAL_PHASE0_PENDING = (
    "value", "threshold", "tolerance", "min_cosine", "median_cosine",
    "num_steps", "chunk_len", "measurement_source",
)


def catalog_protocol_extension(report: dict | None) -> dict | None:
    """The Phase-0 evaluation fields the catalog schema PR will add (RFC §4.3).
    Kept here so the catalog follow-up is a one-call wire-up: when the live
    catalog accepts `value`/`min_cosine`/`protocol`, fold this into
    `_catalog_evaluation`. None when unmeasured."""
    if not report:
        return None
    gb = report.get("gate_b", {})
    ext = {k: gb[k] for k in _EVAL_PHASE0_PENDING if k in gb and gb[k] is not None}
    protocol = gb.get("protocol") or {}
    if isinstance(protocol, dict) and protocol:
        ext["protocol"] = dict(protocol)
    return ext or None


def catalog_evaluation_keys(catalog_path) -> set[str] | None:
    """The property names the TARGET catalog's model.schema.json `evaluation` block
    accepts. Used to feature-detect which Phase-0 fields to emit (RFC §4.3) so the
    fabric↔catalog wire-up is decoupled from merge order: before the schema PR
    lands the catalog lacks these keys and nothing extra is emitted (cross-contract
    stays green); after it lands they flow automatically, no further fabric change.
    None when unreadable — then only the always-accepted baseline is emitted."""
    if not catalog_path:
        return None
    spath = Path(catalog_path) / "schema" / "model.schema.json"
    try:
        schema = json.loads(spath.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    props = (((schema.get("properties") or {}).get("evaluation") or {}).get("properties")) or {}
    return set(props) or None


def _catalog_evaluation(report: dict | None, accepted_keys: set[str] | None = None) -> dict | None:
    """The measured parity signature (from build/<id>/parity-report.json gate_b),
    verbatim, for the catalog `evaluation` field — emitting ONLY the fields the
    live catalog schema accepts (F13: one authority, no rejected additional
    properties). The richer protocol signature lives in the RECIPE
    (gate_b.protocol, written by verify) and the generated scorecard; it reaches
    the catalog via the batched schema follow-up (catalog_protocol_extension).

    None when unmeasured (no numeric core)."""
    if not report:
        return None
    gb = report.get("gate_b", {})
    ev = {k: gb[k] for k in _EVAL_KEYS_ACCEPTED if k in gb and gb[k] is not None}
    # F2: the action lane reports the per-obs min as `min_action_cosine` — the
    # catalog's accepted name is `min_chunk_cosine`. Normalize so the cell
    # comparison is honest.
    if "min_chunk_cosine" not in ev:
        for alias in ("min_action_cosine", "min_cosine"):
            if isinstance(gb.get(alias), (int, float)):
                ev["min_chunk_cosine"] = gb[alias]
                break
    if "mean_chunk_cosine" not in ev and isinstance(gb.get("mean_cosine"), (int, float)):
        ev["mean_chunk_cosine"] = gb["mean_cosine"]
    if not ev.get("metric"):
        return None
    # F6: an evaluation with no numeric core is not a measurement.
    numeric_keys = ("min_chunk_cosine", "mean_chunk_cosine", "margin_gated_match_rate",
                    "argmax_match_rate", "top5_agreement_rate", "max_action_mae",
                    "max_relative_action_mae")
    if not any(isinstance(ev.get(k), (int, float)) for k in numeric_keys):
        return None
    env = gb.get("environment", {})
    # PRIVACY: never surface the publisher's specific hardware/OS build. Report only
    # the generic platform family (the harness emits `platform`/`accelerator`).
    measured_on = env.get("accelerator") or env.get("platform")
    if measured_on and "measured_on" not in ev:
        ev["measured_on"] = str(measured_on)
    # F7 rule 3 (M2): surface any named waiver INTO the catalog evaluation. The live
    # catalog schema doesn't accept a `waivers` field yet (that is the batched
    # protocol-extension PR), so we fold them into the accepted `reason` string —
    # a waivered pass must never reach the catalog looking clean.
    waivers = (gb.get("protocol") or {}).get("waivers")
    if isinstance(waivers, list) and waivers:
        tag = "waivers: " + ", ".join(str(w) for w in waivers)
        ev["reason"] = f"{ev['reason']} [{tag}]" if ev.get("reason") else tag
    # Feature-detected Phase-0 extension (RFC §4.3): emit the richer fields
    # (value / min_cosine / protocol / …) IFF the target catalog schema accepts
    # them. Decoupled from merge order — safe in both states (see
    # catalog_evaluation_keys). Once coreai-catalog's evaluation schema PR lands,
    # `accepted_keys` gains these names and the number + full signature flow to the
    # catalog with no further fabric change.
    if accepted_keys:
        for k, v in (catalog_protocol_extension(report) or {}).items():
            if k in accepted_keys and k not in ev:
                ev[k] = v
    return ev


#: A pass must clear the threshold by at least this absolute margin AND be
#: measured over at least this many observations to be called `high_fidelity`.
#: A bare pass (right at the bar, or underpowered) is `balanced`; a waivered
#: pass is `balanced` too. (RFC F7 rule 2: never collapse pass -> high_fidelity.)
HIGH_FIDELITY_MARGIN = 0.0005
HIGH_FIDELITY_MIN_OBS = 8


def _fidelity_tier(quantization: str, report: dict | None) -> str | None:
    """size (lossy, e.g. int4) / high_fidelity (~lossless) / balanced — from the
    MEASURED gate_b per F7 rule 2 (margin + n_obs + waivers), never a collapse of
    any pass to high_fidelity. Quantization tier is a tiebreaker hint ONLY when
    no measured number exists."""
    gb = (report or {}).get("gate_b", {})
    status = gb.get("status")
    if status == "failed":
        return "size"
    if status == "passed":
        value = gb.get("value")
        threshold = gb.get("threshold")
        n_obs = gb.get("n_obs")
        waivers = (gb.get("protocol") or {}).get("waivers")
        # Margin over the bar: threshold-relative for cosine-style metrics
        # (graph_output_cosine / action_parity / per_token_logit_cosine), absolute
        # for greedy_parity (a match-RATE with no meaningful threshold, e.g. 0.958).
        has_margin = (
            isinstance(value, (int, float))
            and ((isinstance(threshold, (int, float)) and value - threshold >= HIGH_FIDELITY_MARGIN)
                 or (threshold is None and value >= 0.99))
        )
        if (has_margin and isinstance(n_obs, int)
                and n_obs >= HIGH_FIDELITY_MIN_OBS
                and not waivers):
            return "high_fidelity"
        return "balanced"  # a bare/waivered/underpowered pass is NOT high_fidelity
    # Unmeasured (not_run): fall back to the quantization hint, honestly labeled.
    q = str(quantization).lower()
    if q in ("int8", "8bit"):
        return "high_fidelity"
    if q in ("int4", "4bit"):
        return "size"
    return None


def build_model_entry(recipe: Recipe, files: list[dict], *, notes_suffix: str = "",
                      report: dict | None = None, catalog_eval_keys: set[str] | None = None) -> dict:
    catalog_block = recipe.data.get("catalog")
    if not catalog_block:
        raise SystemExit(
            f"error: recipe {recipe.id} has no catalog: block — register needs "
            "name/family/capabilities/modalities to generate the model entry. "
            "Add it to the recipe (see schema/recipe.schema.json)."
        )
    conv = recipe.data["conversion"]
    total_bytes = sum(f.get("size_bytes", 0) for f in files)
    # Runtime facts are declared by the recipe (fabric is the converter, so it is the
    # authoritative source). The catalog audit rejects `unknown` for these four; the
    # recipe schema requires them as real booleans, so they are always present here.
    rf = catalog_block["runtime_facts"]
    # min_os: the recipe may declare it; else default to the apple-core-ai floor
    # (macOS/iOS 27.0, grounded in apple/coreai-models Package.swift).
    min_os = catalog_block.get("min_os", {"macos": "27.0", "ios": "27.0"})
    entry: dict = {
        "id": recipe.id,
        "name": catalog_block["name"],
        "family": catalog_block["family"],
        "source_group": "fabric",
        "source_path": recipe_url(recipe),
        "artifact_ref": recipe.id,
        "capabilities": list(catalog_block["capabilities"]),
        # Architecture/inference traits (moe, mla, …) are orthogonal to the task
        # list; emit them only when the recipe declares them, so the capabilities
        # facet stays a clean controlled vocabulary of what the model DOES.
        **({"traits": list(catalog_block["traits"])} if catalog_block.get("traits") else {}),
        "bundle_kind": catalog_block["bundle_kind"],
        "min_os": {"macos": min_os["macos"], "ios": min_os["ios"]},
        # Upstream repo (org/name) is the discovery dedup key; fabric holds it verbatim.
        "upstream_repo": recipe.data["upstream"]["hf_repo"],
        "modalities": {
            "input": list(catalog_block["modalities"]["input"]),
            "output": list(catalog_block["modalities"]["output"]),
        },
        "artifact": {"format": "aimodel", "availability": "available"},
        "size": {
            "parameters": catalog_block.get("parameters", "not_published"),
            "precision": conv["precision"],
            "quantization": conv["quantization"],
            "artifact_size": _humanize_bytes(total_bytes) if total_bytes else "not_published",
        },
        "runtime": {
            "runtime_name": "apple-core-ai",
            "runner": catalog_block.get("runner", "CoreAIRunner"),
            "stock_runtime": rf["stock_runtime"],
            "custom_kernel": rf["custom_kernel"],
            "patch_required": rf["patch_required"],
            "tokenizer_required": _bool_or_unknown(catalog_block.get("tokenizer_required")),
            "processor_required": _bool_or_unknown(catalog_block.get("processor_required")),
            "aot_required": rf["aot_required"],
        },
        # Device support is unknowable until someone runs the artifact on-device;
        # fabric never asserts it.
        "device_support": {
            "iphone": "unknown",
            "ipad": "unknown",
            "mac": "unknown",
            "mac_only": "unknown",
        },
        "license": {
            "name": recipe.data["upstream"]["license"],
            "commercial_use": commercial_use_for(recipe),
        },
        "status": "needs_review",
        "maturity": "experimental",
        "confidence": "medium",
        "sources": [SOURCE_RECORD_ID],
        "last_verified": today(),
        "notes": (
            f"Converted via coreai-fabric recipe {recipe.id} "
            f"({recipe_url(recipe)}).{notes_suffix}"
        ),
    }
    if catalog_block.get("architecture"):
        entry["architecture"] = catalog_block["architecture"]
    if isinstance(catalog_block.get("streaming"), bool):
        entry["streaming"] = catalog_block["streaming"]
    # C1/E3: carry the MEASURED parity signature + fidelity tier into the catalog
    # (so the number lives as data, not a note).
    # NOTE: variant_group was removed — the catalog schema no longer accepts it.
    # Use source_group instead if grouping is needed in the future.
    tier = _fidelity_tier(conv["quantization"], report)
    if tier:
        entry["size"]["fidelity_tier"] = tier
    evaluation = _catalog_evaluation(report, catalog_eval_keys)
    if evaluation:
        entry["evaluation"] = evaluation
    # C4: emit a typed io_contract so a fabric model is as agent-ready as the
    # official ones (not just a bundle). None for bundle kinds fabric can't yet
    # describe truthfully — the catalog's every-fabric-model-has-io_contract
    # test then FORCES the contract to be added, never silently skipped.
    io = _io_contract(catalog_block, recipe.data["upstream"]["hf_repo"])
    if io:
        entry["io_contract"] = io
    return entry


def _io_contract(catalog_block: dict, upstream_repo: str) -> dict | None:
    """A typed integration contract for the bundle kinds fabric produces and
    verifies today. LLM = the KV-cache chat asset (text->text, stateful streaming,
    CoreAILanguageModel). ACTION = a VLA/robot policy (obs in, action chunk out,
    NON-stateful, host-driven sampler). Other kinds return None until fabric emits
    + verifies them — the catalog's fabric-io_contract test then forces coverage."""
    kind = catalog_block.get("bundle_kind")
    if kind == "action":
        # A policy is NOT chat: honest non-stateful, non-streaming; obs modalities
        # in, a continuous action chunk out via a host-driven sampler loop.
        inputs = [
            {"name": m, "modality": m}
            for m in (catalog_block.get("modalities", {}) or {}).get("input", [])
        ] or [{"name": "observation", "modality": "image"}]
        contract = {
            "entrypoint": {
                "framework": "CoreAIRunner",
                "type": "CoreAIRunner",
                "init_pattern": "load the encode + denoise_step graphs; host drives the N-step sampler + un-normalization",
            },
            "inputs": inputs,
            "outputs": [{
                "name": "action_chunk",
                "swift_type": "[[Float]]",
                "decoding": {
                    "detokenization": "host-driven flow-matching sampler (N steps); "
                    "un-normalize the action chunk with norm_stats.json",
                },
            }],
            "session": {"stateful": False, "streaming": False},
        }
        files = {}
        if catalog_block.get("tokenizer_required"):
            files["tokenizer_ref"] = "tokenizer"
        if catalog_block.get("processor_required"):
            files["processor_ref"] = "norm_stats.json"
        if files:
            contract["files"] = files
        return contract
    if kind != "llm":
        return None
    prompt_input = {"name": "prompt", "modality": "text", "swift_type": "String"}
    ctx = catalog_block.get("context_length")
    if isinstance(ctx, int):
        prompt_input["constraints"] = {"max_context": ctx}
    contract = {
        "entrypoint": {
            "framework": "CoreAILanguageModels",
            "type": "CoreAILanguageModel",
            "init_pattern": "let model = try await CoreAILanguageModel(resourcesAt: bundleURL)",
        },
        "inputs": [prompt_input],
        "outputs": [{
            "name": "response",
            "swift_type": "String",
            "decoding": {
                "detokenization": f"embedded tokenizer ({upstream_repo}); "
                "LanguageModelSession.respond(to:) returns detokenized text",
            },
        }],
        "session": {"stateful": True, "streaming": True},
    }
    if catalog_block.get("tokenizer_required"):
        contract["files"] = {"tokenizer_ref": "macos/tokenizer"}
    return contract


def build_artifact_entry(recipe: Recipe, files: list[dict], tool_version: str | None) -> dict:
    published = recipe.data.get("published")
    if not published:
        raise SystemExit(
            f"error: recipe {recipe.id} has no published block — run "
            f"`coreai-fabric publish {recipe.id}` first (register indexes "
            "published artifacts only)."
        )
    hf_repo = published["hf_repo"]
    owner, repo = hf_repo.split("/", 1)
    conv = recipe.data["conversion"]
    pub = recipe.data.get("publish", {})
    provenance: dict = {
        "converted_by": {
            "tool": conv["tool"],
            "version": tool_version or "unknown",
            "recipe_url": recipe_url(recipe),
        },
        "recipe_source": "fabric",
    }
    fmt = (recipe.data.get("expected") or {}).get("format_version")
    if fmt:
        provenance["format_version"] = fmt
    entry: dict = {
        "id": recipe.id,
        # Per the shared field contract the artifact group enum is unchanged
        # (zoo/official/external/unknown) while the MODEL gains source_group
        # fabric; external is the honest artifact group for an independent
        # conversion. Known caveat: catalog audit category 2 compares
        # artifact.group with model.source_group, so the catalog-side audit
        # must learn the fabric<->external pairing before a register PR can
        # pass its local audit run (register aborts safely until then).
        "group": "external",
        # No github block: fabric provenance is huggingface-native. The catalog
        # artifact schema allows this via anyOf(github, huggingface) — no more
        # fabricated GitHub coordinates for HF-only conversions.
        "huggingface": {
            "owner": owner,
            "repo": repo,
            "url": f"https://huggingface.co/{hf_repo}",
            "revision": published["revision"],
            "files": files,
        },
        "provenance": provenance,
        "officiality": {
            "apple_export_recipe": False,
            "apple_hosted_artifact": False,
            "community_packaged": True,
        },
    }
    # C3: a variant tier lives under `<variant>/` in the shared repo — scope the
    # host path so int4/ and int8/ don't collide at the artifact level. Follow
    # the catalog's existing convention (gemma-4-e2b-vision, efficientsam3):
    # `path` is the browsable `tree/main/<subdir>` and `url` includes it, so the
    # audit's `url == base + "/" + path` integrity check holds.
    if pub.get("variant"):
        subpath = f"tree/main/{pub['variant']}"
        entry["huggingface"]["path"] = subpath
        entry["huggingface"]["url"] = f"https://huggingface.co/{hf_repo}/{subpath}"
    # C3: the community mirror as machine-readable data (source <-> mirror), so
    # the catalog's neutral-index claim is real. Same bytes, alternative host.
    if pub.get("mirror_namespace"):
        entry["mirrors"] = [{
            "owner": pub["mirror_namespace"],
            "repo": pub["repo_name"],
            "url": f"https://huggingface.co/{pub['mirror_namespace']}/{pub['repo_name']}",
            "revision": published["revision"],
        }]
    return entry


def build_source_record() -> dict:
    """sources.yaml record the catalog needs so `sources: [coreai-fabric]`
    cross-references resolve. Shape follows the catalog's
    schema/source.schema.json (strict, additionalProperties: false)."""
    return {
        "id": SOURCE_RECORD_ID,
        "title": "kevinqz/coreai-fabric",
        "type": "github_repository",
        "url": FABRIC_REPO_URL,
        "owner": "kevinqz",
        "repo": "coreai-fabric",
        "trust": "project_primary",
        "volatility": "medium",
        "last_checked": today(),
        "notes": "First-party conversion pipeline: recipes in, provenance-verified .aimodel out.",
    }


def validate_against_catalog_schemas(
    catalog_path: Path, model_entry: dict, artifact_entry: dict
) -> list[str]:
    """Validate generated entries against the catalog clone's own schemas.
    Returns aggregated error strings (empty = valid)."""
    errors: list[str] = []
    schema_dir = catalog_path / "schema"
    for label, schema_file, entry in (
        ("model", "model.schema.json", model_entry),
        ("artifact", "artifact.schema.json", artifact_entry),
    ):
        spath = schema_dir / schema_file
        if not spath.is_file():
            errors.append(f"{label}: catalog schema missing at {spath}")
            continue
        schema = json.loads(spath.read_text())
        validator = Draft202012Validator(schema)
        for e in sorted(validator.iter_errors(entry), key=lambda e: list(e.absolute_path)):
            path = ".".join(str(p) for p in e.absolute_path) or "<root>"
            errors.append(f"{label}.{path}: {e.message}")
    # Actionable contract hint when the clone predates the shared field contract.
    if any("'fabric'" in e or "fabric" in e and "enum" in e for e in errors):
        errors.append(
            "hint: the catalog clone's schemas may predate the fabric field "
            "contract (source_group 'fabric', optional github, huggingface "
            "revision/files, provenance). Update the clone."
        )
    return errors


def _resolve_published_digests(recipe: Recipe) -> list[dict]:
    from . import hf

    published = recipe.data.get("published")
    if not published:
        raise SystemExit(
            f"error: recipe {recipe.id} is not published (status: {recipe.status}) — "
            f"run `coreai-fabric publish {recipe.id}` first."
        )
    try:
        return hf.file_digests(published["hf_repo"], published["revision"])
    except hf.HFError as exc:
        raise SystemExit(f"error: cannot fetch published file digests: {exc}") from exc


def _tool_version_from_manifest(root: Path, recipe: Recipe) -> str | None:
    mpath = manifest_path(root, recipe)
    if mpath.is_file():
        return json.loads(mpath.read_text()).get("tool_version")
    return None


def _load_parity_report(root: Path, recipe: Recipe) -> dict | None:
    """Load the parity report, and ensure the durable protocol signature (RFC
    Phase 0, written into the recipe by verify) is present on gate_b even when
    the gitignored build/<id>/parity-report.json is stale or absent. The recipe
    is the provenance record; the report is the transient measurement."""
    rpath = parity_report_path(root, recipe)
    report = None
    if rpath.is_file():
        try:
            report = json.loads(rpath.read_text())
        except json.JSONDecodeError:
            report = None
    # The recipe carries the durable protocol block + the canonical threshold;
    # fold it onto the report's gate_b so register always sees the full signature.
    recipe_gate_b = (recipe.data.get("parity") or {}).get("gate_b") or {}
    recipe_protocol = recipe_gate_b.get("protocol")
    recipe_measured = recipe_gate_b.get("measured")
    if report is None:
        # RFC F6: the gitignored build report is gone (fresh clone / CI), but the
        # recipe carries the DURABLE measured result + protocol (written by verify).
        # Synthesize a report from them so _catalog_evaluation / _fidelity_tier run
        # and the catalog gets the number — the exact fresh-clone gap the audit found.
        if not (recipe_measured or recipe_protocol):
            return None
        gb = {"metric": recipe_gate_b.get("metric")}
        if isinstance(recipe_measured, dict):
            gb.update({k: v for k, v in recipe_measured.items() if v is not None})
        gb.setdefault("status", (recipe_measured or {}).get("status", "not_run"))
        report = {"gate_a": {"status": "passed"}, "gate_b": gb}
    gb = report.setdefault("gate_b", {})
    if isinstance(recipe_protocol, dict) and recipe_protocol:
        gb["protocol"] = dict(recipe_protocol)
    # Fold the durable measured result + the canonical gate definition onto the
    # report's gate_b when the (transient) report lacks them.
    if isinstance(recipe_measured, dict):
        for k, v in recipe_measured.items():
            if k not in gb and v is not None:
                gb[k] = v
    for k in ("threshold", "tolerance"):
        if k not in gb and k in recipe_gate_b:
            gb[k] = recipe_gate_b[k]
    return report


def _notes_suffix_from_report(report: dict | None) -> str:
    if not report:
        return ""
    return (
        f" Parity: gate A {report['gate_a']['status']}, "
        f"gate B {report['gate_b']['status']}."
    )


def cmd_register(args) -> int:
    root = find_root()
    recipe = find_recipe(args.id, root)

    # --mark-merged: the human merged the catalog PR; flip published -> registered.
    if getattr(args, "mark_merged", False):
        pr = recipe.data.get("catalog_pr")
        if not pr:
            err(f"recipe {recipe.id} has no catalog_pr — nothing to mark merged "
                "(run register to open the PR first)")
            return 1
        recipe.data["status"] = "registered"
        write_yaml(recipe.path, recipe.data)
        ok(f"recipe {recipe.id} status -> registered (PR {pr} merged)")
        return 0

    files = _resolve_published_digests(recipe)
    tool_version = _tool_version_from_manifest(root, recipe)
    report = _load_parity_report(root, recipe)
    catalog_path = Path(args.catalog_path).resolve() if args.catalog_path else None
    # Feature-detect which evaluation fields the target catalog accepts (RFC §4.3).
    eval_keys = catalog_evaluation_keys(catalog_path)
    model_entry = build_model_entry(
        recipe, files, notes_suffix=_notes_suffix_from_report(report), report=report,
        catalog_eval_keys=eval_keys,
    )
    artifact_entry = build_artifact_entry(recipe, files, tool_version)
    source_record = build_source_record()

    if catalog_path:
        errors = validate_against_catalog_schemas(catalog_path, model_entry, artifact_entry)
        if errors:
            err(f"generated entries do not validate against {catalog_path}/schema:")
            for line in errors:
                print(f"  - {line}", file=sys.stderr)
            return 1
        ok("generated entries validate against the catalog schemas")

    if args.dry_run or not catalog_path:
        if not catalog_path:
            warn("no --catalog-path given: schema validation skipped, printing YAML only")
        print("# --- append to catalog.yaml under models: ---")
        print(dump_yaml({"models": [model_entry]}))
        print("# --- append to artifacts.yaml under artifacts: (and bump metadata.count) ---")
        print(dump_yaml({"artifacts": [artifact_entry]}))
        print("# --- ensure this record exists in sources.yaml under sources: ---")
        print(dump_yaml({"sources": [source_record]}))
        return 0

    return _apply_and_open_pr(root, recipe, catalog_path, model_entry, artifact_entry, source_record, args)


def _apply_and_open_pr(
    root: Path,
    recipe: Recipe,
    catalog_path: Path,
    model_entry: dict,
    artifact_entry: dict,
    source_record: dict,
    args,
) -> int:
    from . import CATALOG_REPO

    branch = f"fabric/add-{recipe.id}"

    def git(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(catalog_path), *cmd],
                              capture_output=True, text=True, check=check)

    status = git("status", "--porcelain").stdout.strip()
    if status:
        err(f"catalog clone at {catalog_path} has uncommitted changes — refusing to branch over them")
        return 1

    git("checkout", "-b", branch)

    # Text-append entries (preserves existing formatting/diff hygiene; the
    # catalog files end with their entry lists).
    _append_entry(catalog_path / "catalog.yaml", {"models": [model_entry]})
    _append_entry(catalog_path / "artifacts.yaml", {"artifacts": [artifact_entry]})
    _bump_artifact_count(catalog_path / "artifacts.yaml")
    _ensure_source_record(catalog_path / "sources.yaml", source_record)

    # E5: appending a model makes the public counts (README, llms.txt, agent.json,
    # site, ...) stale. Bump them to canonical BEFORE the gate, so the count-sync
    # check passes and the PR arrives green instead of tripping catalog CI. The
    # gate below re-verifies, so a --fix that misses a surface still fails loud.
    cc = catalog_path / "scripts" / "check_counts.py"
    if cc.exists():
        subprocess.run([sys.executable, str(cc), "--fix"], cwd=catalog_path,
                       capture_output=True, text=True)

    # Replay the catalog's OWN CI gate locally so the PR arrives green. Mirrors
    # .github/workflows/validate.yml, not a 3-script subset — otherwise a PR can
    # pass here and still fail catalog CI. Missing scripts (older checkout) are
    # skipped with a warning.
    gates = [
        ([sys.executable, "scripts/validate.py"], "validate", "scripts/validate.py"),
        ([sys.executable, "scripts/audit.py"], "audit", "scripts/audit.py"),
        ([sys.executable, "scripts/generate.py"], "generate", "scripts/generate.py"),
        ([sys.executable, "scripts/check_counts.py"], "check_counts", "scripts/check_counts.py"),
        ([sys.executable, "scripts/doc_test.py"], "doc_test", "scripts/doc_test.py"),
        ([sys.executable, "scripts/generate_templates.py", "--check"], "templates --check", "scripts/generate_templates.py"),
        ([sys.executable, "scripts/injection_lint.py"], "injection_lint", "scripts/injection_lint.py"),
        ([sys.executable, "-m", "pytest", "tests/test_p1_iocontract.py", "-q"], "io_contract tests", "tests/test_p1_iocontract.py"),
    ]
    for argv, label, needs in gates:
        if not (catalog_path / needs).exists():
            warn(f"catalog has no {needs}; skipping gate '{label}'")
            continue
        proc = subprocess.run(argv, cwd=catalog_path, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-6:])
            err(f"catalog gate '{label}' failed:\n{tail}")
            err(f"branch {branch} left in place at {catalog_path} for inspection; no PR opened")
            return 1
        print(f"  catalog gate {label}: ok")

    git("add", "-A")
    git("commit", "-m", f"feat: add {recipe.id} via coreai-fabric\n\nRecipe: {recipe_url(recipe)}")

    # Third-party publishers have no push access to the canonical catalog. Try a
    # direct push; on failure, fork and push to the fork, then open a cross-repo PR.
    head_ref = branch
    push = git("push", "-u", "origin", branch, check=False)
    if push.returncode != 0:
        warn("no push access to the catalog's origin — forking (third-party publisher path)")
        subprocess.run(["gh", "repo", "fork", CATALOG_REPO, "--remote=false", "--clone=false"],
                       cwd=catalog_path, capture_output=True, text=True)
        who = subprocess.run(["gh", "api", "user", "--jq", ".login"],
                             capture_output=True, text=True)
        fork_owner = who.stdout.strip()
        if not fork_owner:
            err(f"could not resolve your GitHub login for the fork:\n{who.stderr}")
            return 1
        fork_url = f"https://github.com/{fork_owner}/{CATALOG_REPO.split('/')[-1]}.git"
        git("remote", "add", "fork", fork_url, check=False)
        fpush = git("push", "-u", "fork", branch, check=False)
        if fpush.returncode != 0:
            err(f"push to fork {fork_owner} failed:\n{fpush.stderr}")
            return 1
        head_ref = f"{fork_owner}:{branch}"

    pr = subprocess.run(
        [
            "gh", "pr", "create",
            "--repo", CATALOG_REPO,
            "--head", head_ref,
            "--title", f"Add {recipe.id} (fabric conversion)",
            "--body", _pr_body(recipe),
        ],
        cwd=catalog_path, capture_output=True, text=True,
    )
    if pr.returncode != 0:
        err(f"gh pr create failed:\n{pr.stderr}")
        return 1
    pr_url = pr.stdout.strip()
    print(pr_url)

    # Honest status: the PR is OPEN, not merged. Indexing happens on merge, so
    # status stays `published` and we record the PR URL. `register --mark-merged`
    # flips to `registered` once a human merges.
    recipe.data["catalog_pr"] = pr_url
    write_yaml(recipe.path, recipe.data)
    ok(f"catalog PR opened: {pr_url}")
    ok(f"recipe {recipe.id} stays status='published' until the PR merges "
       f"(then: coreai-fabric register --mark-merged {recipe.id})")
    return 0


def _append_entry(path: Path, wrapper: dict) -> None:
    # Dump the single-entry wrapper and strip the top-level key line, leaving
    # a correctly-indented "- id: ..." list item to append.
    text = dump_yaml(wrapper)
    lines = text.splitlines()
    entry_text = "\n".join(lines[1:]) + "\n"
    existing = path.read_text()
    if not existing.endswith("\n"):
        existing += "\n"
    path.write_text(existing + entry_text)


def _bump_artifact_count(path: Path) -> None:
    text = path.read_text()
    match = re.search(r"^(\s*count:\s*)(\d+)\s*$", text, flags=re.MULTILINE)
    if not match:
        warn("artifacts.yaml has no metadata count: line to bump")
        return
    new = f"{match.group(1)}{int(match.group(2)) + 1}"
    path.write_text(text[: match.start()] + new + text[match.end():])


def _ensure_source_record(path: Path, record: dict) -> None:
    import yaml

    data = yaml.safe_load(path.read_text()) or {}
    ids = {s.get("id") for s in data.get("sources", [])}
    if record["id"] in ids:
        return
    _append_entry(path, {"sources": [record]})


def _pr_body(recipe: Recipe) -> str:
    published = recipe.data["published"]
    return (
        f"Adds `{recipe.id}`, converted and published via coreai-fabric.\n\n"
        f"- Recipe: {recipe_url(recipe)}\n"
        f"- Upstream: https://huggingface.co/{recipe.data['upstream']['hf_repo']}"
        f" (license: {recipe.data['upstream']['license']})\n"
        f"- Published artifact: https://huggingface.co/{published['hf_repo']}"
        f" @ `{published['revision']}`\n"
        f"- Provenance: `source_group: fabric`, `provenance.recipe_source: fabric`,"
        f" huggingface-only artifact block with pinned revision + per-file sha256\n\n"
        "Generated by `coreai-fabric register`. Parity and conversion reports "
        "ship inside the published HF repo.\n\n"
        "🤖 Generated with [coreai-fabric](https://github.com/kevinqz/coreai-fabric)\n"
    )
