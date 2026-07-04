"""`coreai-fabric publish` — upload the build output to the publisher's own
HF namespace and record the result back into the recipe.

Fabric never hosts weights: the target is always the publisher's namespace
(publish.hf_target_namespace in the recipe). Requires huggingface_hub
(pip install "coreai-fabric[hf]") and a logged-in token (`hf auth login`).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .convert import bundle_path, manifest_path
from .recipes import find_recipe, triage_license
from .util import err, find_root, ok, today, warn, write_yaml
from .verify import parity_report_path

#: Candidate license/notice filenames to mirror from the upstream repo so the
#: published derivative satisfies Apache-2.0 §4(a)/(d) and similar terms.
UPSTREAM_LICENSE_FILES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE", "NOTICE.txt")

#: Absolute-path markers that must never reach a public repo (privacy leak).
_LOCAL_PATH_MARKERS = ("/Users/", "/home/")


def fetch_upstream_license(hf_repo: str, revision: str | None, staging: Path) -> list[str]:
    """Download the upstream LICENSE/NOTICE into staging (Apache-2.0 §4(a)/(d):
    a redistribution must ship a copy of the license + retain the NOTICE). The
    caller treats an absent-but-required license as a hard publish failure —
    fabric never redistributes weights without their license text."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    written: list[str] = []
    for name in UPSTREAM_LICENSE_FILES:
        try:
            path = hf_hub_download(hf_repo, name, revision=revision, repo_type="model")
        except EntryNotFoundError:
            continue
        except Exception:  # noqa: BLE001 — network/other: try the next candidate
            continue
        # Prefix so it never collides with anything fabric writes.
        dest = staging / (name if name.startswith(("LICENSE", "NOTICE")) else f"upstream-{name}")
        dest.write_text(Path(path).read_text())
        written.append(dest.name)
    return written


def copyright_holder_from_license(staging: Path) -> str | None:
    """Best-effort: the copyright holder to retain per Apache-2.0 §4(c), read
    from the fetched LICENSE (e.g. 'Copyright 2024 Alibaba Cloud'). None if not
    found — the card then omits the attribution line rather than inventing one."""
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE"):
        f = staging / name
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            m = re.search(r"copyright\s*(?:\(c\)|©)?\s*(\d{4}.*|.*\d{4}.*)", line, re.I)
            if m and m.group(1).strip() and "[" not in m.group(1):  # skip template "[yyyy]"
                return m.group(1).strip().rstrip(".")
    return None


def sanitize_manifest(manifest: dict, root: Path) -> dict:
    """A PUBLIC, reproducible-by-anyone copy of the conversion manifest: drop
    the host-specific absolute `tool_path` and relativize any local path in
    `command[]` (which otherwise leaks the OS username + local dir layout into a
    permanent public git history)."""
    m = {k: v for k, v in manifest.items() if k != "tool_path"}
    root_str, home = str(root), str(Path.home())
    if isinstance(m.get("command"), list) and m["command"]:
        cmd = [os.path.basename(str(m["command"][0]))]
        for tok in m["command"][1:]:
            cmd.append(str(tok).replace(root_str, ".").replace(home, "~"))
        m["command"] = cmd
    return m


def assert_no_local_paths(staging: Path) -> list[str]:
    """Last-line guard before a public upload: any staged text file still
    carrying an absolute local path is a leak. Returns the offending filenames."""
    markers = (*_LOCAL_PATH_MARKERS, str(Path.home()))
    leaks: list[str] = []
    for p in sorted(staging.rglob("*")):
        if not p.is_file():
            continue
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if any(mk in text for mk in markers):
            leaks.append(str(p.relative_to(staging)))
    return leaks


def sibling_variants(recipe, root: Path) -> list:
    """All recipes publishing into the SAME repo (namespace+repo_name) that
    declare a `publish.variant` — the tiers of a multi-variant repo. Returns
    [(variant, recipe, parity_report_or_None)] sorted by variant name."""
    from .recipes import load_all_recipes
    pub = recipe.data.get("publish", {})
    key = (pub.get("hf_target_namespace"), pub.get("repo_name"))
    out = []
    for r in load_all_recipes(root):
        rp = r.data.get("publish", {})
        if rp.get("variant") and (rp.get("hf_target_namespace"), rp.get("repo_name")) == key:
            rpath = parity_report_path(root, r)
            report = None
            if rpath.is_file():
                try:
                    report = json.loads(rpath.read_text())
                except json.JSONDecodeError:
                    pass
            out.append((rp["variant"], r, report))
    return sorted(out, key=lambda x: x[0])


def _variants_table(recipe, root: Path) -> str:
    """A 'Quantization variants' comparison table across the repo's tiers —
    each variant's quant, on-disk size, and measured greedy_parity (or pending)."""
    sibs = sibling_variants(recipe, root)
    if len(sibs) < 2:
        return ""
    rows = []
    for variant, r, report in sibs:
        conv = r.data["conversion"]
        quant = conv.get("quantization", "—")
        main = bundle_path(root, r) / "main.mlirb"
        size = _human_size(main.stat().st_size) if main.is_file() else "—"
        gb = (report or {}).get("gate_b", {}) if report else {}
        if isinstance(gb.get("argmax_match_rate"), (int, float)):
            argmax = f"{round(100*gb['argmax_match_rate'],1)}%"
            top5 = f"{round(100*gb.get('top5_agreement_rate',0),1)}%" if isinstance(gb.get("top5_agreement_rate"), (int, float)) else "—"
        else:
            argmax = top5 = "pending"
        rows.append(f"| `{variant}/` | {quant} | {size} | {argmax} | {top5} |")
    return (
        "\n## Quantization variants\n\n"
        "This repo ships multiple tiers of the same conversion. Greedy fidelity is "
        "per-token argmax agreement vs the fp16 reference (see Evaluation) — the "
        "numbers below are measured, or `pending` until you run the parity runner.\n\n"
        "| Variant | Quant | On-disk | Greedy argmax | Top-5 |\n"
        "|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
        "**int4** is the size-optimized tier; **int8** is the high-fidelity tier. "
        "Pick by your size/quality budget — the measured numbers above tell you the "
        "fidelity cost, so you never guess.\n"
    )


def _human_size(num_bytes: int) -> str:
    step = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if step < 1024 or unit == "GB":
            return f"{step:.0f} {unit}" if unit != "GB" else f"{step:.1f} {unit}"
        step /= 1024
    return f"{num_bytes} B"


def _bundle_metadata(root: Path, recipe) -> dict:
    """The Apple-produced .aimodel metadata.json (assetVersion, producer, real
    quantization if present) — the AUTHORITATIVE description of the shipped
    asset, preferred over the recipe's documentation-only fields."""
    f = bundle_path(root, recipe) / "metadata.json"
    if f.is_file():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def render_model_card(root: Path, recipe, manifest: dict, report: dict,
                      *, copyright_holder: str | None = None,
                      collection_url: str | None = None) -> str:
    template = (root / "templates" / "model-card.md").read_text()
    upstream = recipe.data["upstream"]
    conv = recipe.data["conversion"]
    gate_b = report["gate_b"]
    from . import FABRIC_REPO_URL

    catalog_block = recipe.data.get("catalog") or {}
    publish_cfg = recipe.data.get("publish") or {}
    repo_id = f"{publish_cfg.get('hf_target_namespace')}/{publish_cfg.get('repo_name')}"

    # SotA discoverability metadata. base_model_relation: a CoreAI export is a
    # `quantized` derivative when the preset compresses weights — NOT `finetune`
    # (HF's default, and what the community's own cards wrongly show). Omit it
    # for an uncompressed (`none`) export rather than mislabel.
    quant = str(conv.get("quantization", "none")).strip()
    is_quantized = quant.lower() not in ("", "none")
    base_model_relation_line = "base_model_relation: quantized\n" if is_quantized else ""
    # Real quantization label + caveat so the card NEVER mislabels the tier (an
    # int8 asset must not be called "4-bit", and vice-versa — session learning).
    quant_label = {"4bit": "4-bit", "int8": "int8", "none": "uncompressed (fp16)"}.get(
        quant.lower(), quant)
    quant_caveat = {
        "4bit": "4-bit quantized — the size-optimized tier; expect small quality deltas vs. fp16",
        "int8": "int8 quantized — the high-fidelity tier, near-lossless vs. fp16",
        "none": "uncompressed (fp16) — full precision",
    }.get(quant.lower(), f"{quant_label} quantized")

    # Tags = accurate descriptors that also happen to be the facets the whole
    # Core AI ecosystem is found under: both spellings (coreai/core-ai) to bridge
    # the community's split; coreml (Core AI is Core ML's successor lineage);
    # the device/runtime facets (iphone, apple-silicon, metal) — all true of a
    # 320 MB on-device asset; and the task + bundle_kind + quantization.
    tags = ["coreai", "core-ai", "coreai-fabric", "aimodel", "coreml",
            "apple", "apple-silicon", "on-device", "iphone", "metal"]
    if upstream.get("pipeline_tag"):
        tags.append(str(upstream["pipeline_tag"]))
    if catalog_block.get("bundle_kind"):
        tags.append(str(catalog_block["bundle_kind"]))
    if is_quantized:
        tags.append(quant.lower())
    seen: set[str] = set()
    tags_block = "".join(f"- {t}\n" for t in tags if not (t in seen or seen.add(t)))

    # Curated example prompts (real chat turns this model handles well) — genuine
    # usage documentation, rendered on the page as the `widget:` examples.
    widget_prompts = catalog_block.get("example_prompts") or (
        ["Explain on-device AI in one sentence.",
         "Write a haiku about Apple Silicon."]
        if (catalog_block.get("bundle_kind") == "llm") else [])
    widget_block = ""
    if widget_prompts:
        widget_block = "widget:\n" + "".join(f'- text: "{p}"\n' for p in widget_prompts)

    # Optional `language:` frontmatter (drives HF's language facet) — only when
    # the recipe declares it (never invented).
    langs = catalog_block.get("languages") or upstream.get("languages")
    language_block = ""
    if langs:
        language_block = "language:\n" + "".join(f"- {l}\n" for l in langs)

    # ---- Model facts (all already known from the validated build) ----
    meta = _bundle_metadata(root, recipe)
    min_os = catalog_block.get("min_os") or {}
    min_os_str = f"macOS {min_os.get('macos', '27.0')}+ / iOS {min_os.get('ios', '27.0')}+"
    ctx = catalog_block.get("context_length") or conv.get("max_context_length") or "—"
    bundle_bytes = 0
    main = bundle_path(root, recipe) / "main.mlirb"
    if main.is_file():
        bundle_bytes = main.stat().st_size
    size_str = _human_size(bundle_bytes) if bundle_bytes else "—"
    facts_rows = [
        ("Parameters", catalog_block.get("parameters", "—")),
        ("Architecture", catalog_block.get("architecture", "—")),
        ("Capabilities", ", ".join(catalog_block.get("capabilities", [])) or "—"),
        ("Quantization / precision", f"{conv.get('quantization', '—')} / {conv.get('precision', '—')}"),
        ("Context length", str(ctx)),
        ("On-disk size", size_str),
        ("Asset kind", "stateful KV-cache chat bundle; embedded tokenizer + chat template"),
        ("assetVersion", meta.get("assetVersion", "—")),
    ]
    facts_block = "".join(f"| {k} | {v} |\n" for k, v in facts_rows)
    variants_block = _variants_table(recipe, root)

    # ---- Evaluation block: real greedy-parity when measured, honest pending otherwise ----
    gb = report.get("gate_b", {})
    if gb.get("metric") == "greedy_parity" and isinstance(gb.get("value"), (int, float)):
        compared = gb.get("compared")
        env = gb.get("environment", {})
        dev = env.get("chip") or "Apple Silicon"
        ref = {"float16": "fp16", "float32": "fp32"}.get(gb.get("reference_dtype"), gb.get("reference_dtype", "fp16"))

        def _pct(k):
            v = gb.get(k)
            return f"{round(100 * v, 1)}%" if isinstance(v, (int, float)) else "—"
        gated = _pct("value")  # margin_gated_match_rate (the primary)
        ci = gb.get("margin_gated_ci95")
        ci_str = f" (95% CI {round(100*ci[0],1)}–{round(100*ci[1],1)}%)" if isinstance(ci, list) and len(ci) == 2 else ""
        evaluation_block = (
            f"- **Gate B — greedy fidelity vs the {ref} reference: {gated} margin-gated"
            f"{ci_str}** · {_pct('argmax_match_rate')} exact-argmax · {_pct('top5_agreement_rate')} "
            f"top-5, over {compared} teacher-forced tokens, measured on-device ({dev}, "
            f"{env.get('os','')}). Margin-gated forgives near-tie flips (where even the "
            "reference flips on rounding noise). This is *fidelity to the reference*, not a "
            "quality verdict. Reproduce with `coreai-fabric verify` + the parity runner "
            "(`parity-report.json`)."
        )
        sm = gb.get("sample")
        if sm and sm.get("prompt"):
            evaluation_block += (
                f"\n  - Sample — prompt `{sm['prompt']}` → asset: "
                f"`{sm.get('asset_argmax','').strip()}`"
            )
    else:
        evaluation_block = (
            f"- **Gate B (numeric accuracy): {gb.get('status', 'not_run')}.** Task-accuracy "
            "evaluation (e.g. tinyMMLU) is pending *upstream*: Apple's `coreai.llm.eval` is a "
            "stub in coreai-models 0.1.0 that cannot score a stateful KV-cache asset. Greedy "
            "fidelity vs fp32 can be measured on-device via the parity runner. fabric never "
            "fakes a parity number."
        )

    # ---- Attribution (Apache-2.0 §4(b)/(c)) ----
    holder = copyright_holder or upstream.get("copyright_holder")
    attribution = (
        f"Weights © {holder}, " if holder else "Weights "
    ) + (
        f"licensed **{upstream['license']}** — see the bundled `LICENSE`."
    )

    # ---- Links + mirror banner ----
    mirror_ns = publish_cfg.get("mirror_namespace")
    mirror_line = (
        f"> **Canonical:** [`{repo_id}`](https://huggingface.co/{repo_id}) — source of truth. "
        + (f"**Mirror:** [`{mirror_ns}/{publish_cfg.get('repo_name')}`](https://huggingface.co/{mirror_ns}/{publish_cfg.get('repo_name')})."
           if mirror_ns else "")
    )
    collection_link = f"- [HF Collection]({collection_url})\n" if collection_url else ""

    return template.format(
        license=upstream["license"],
        upstream_hf_repo=upstream["hf_repo"],
        upstream_revision=manifest.get("input", {}).get("revision")
        or upstream.get("revision", "unpinned"),
        pipeline_tag=upstream.get("pipeline_tag", ""),
        base_model_relation_line=base_model_relation_line,
        language_block=language_block,
        widget_block=widget_block,
        tags_block=tags_block,
        name=catalog_block.get("name", recipe.id),
        recipe_id=recipe.id,
        repo_id=repo_id,
        mirror_line=mirror_line,
        facts_block=facts_block,
        variants_block=variants_block,
        quant_label=quant_label,
        quant_caveat=quant_caveat,
        min_os=min_os_str,
        evaluation_block=evaluation_block,
        attribution=attribution,
        collection_link=collection_link,
        recipe_url=f"{FABRIC_REPO_URL}/blob/main/recipes/{recipe.id}.yaml",
        tool=manifest.get("tool", conv["tool"]),
        tool_version=manifest.get("tool_version") or "(version not reported)",
        precision=conv["precision"],
        quantization=conv["quantization"],
        date=today(),
        gate_a_status=report["gate_a"]["status"],
        gate_b_metric=gate_b["metric"],
        gate_b_status=gate_b["status"],
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
    gate_b_status = report["gate_b"]["status"]
    if gate_b_status == "failed":
        # A MEASURED failure is different from not-yet-measured — do not let it
        # ship on the same flag as `not_run`. This is the int4-lossy-on-Qwen case
        # (measured 0.81 < 0.9): publishable only as an explicit size tier.
        val = report["gate_b"].get("argmax_match_rate") or report["gate_b"].get("value")
        pct = f" (measured {round(100*val,1)}% argmax)" if isinstance(val, (int, float)) else ""
        if not getattr(args, "publish_known_lossy_size_tier", False):
            err(f"Gate B FAILED{pct} — this asset does NOT meet the fidelity bar. "
                "It is not merely unmeasured; it is measured-lossy. Publish it only "
                "as an explicit size-optimized tier with --publish-known-lossy-size-tier "
                "(the card will carry the measured number + a size-tier caveat). For "
                "high fidelity, use the int8 lane (conversion.compression_config).")
            return 1
        warn(f"publishing a MEASURED-LOSSY size tier{pct} — the card states so honestly")
    elif gate_b_status != "passed":  # not_run
        if args.allow_unverified_parity:
            warn(f"publishing with gate B {gate_b_status} — the model card will say so")
        else:
            err("Gate B has not run (no parity number). Install the parity runner "
                "and re-verify to MEASURE fidelity, or pass --allow-unverified-parity "
                "to publish a structurally-valid but numerically-unproven artifact.")
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
    upstream = recipe.data["upstream"]
    publish_cfg = recipe.data["publish"]
    repo_id = f"{publish_cfg['hf_target_namespace']}/{publish_cfg['repo_name']}"
    revision_pin = manifest.get("input", {}).get("revision") or upstream.get("revision")

    # Assemble the staging tree (card + upstream LICENSE/NOTICE + sanitized
    # reproduce-manifest + parity report). The upstream LICENSE is REQUIRED —
    # redistributing weights without it breaches Apache-2.0 §4(a)/(d).
    staging = root / "build" / recipe.id / "publish-staging"
    if staging.exists():
        import shutil
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        license_files = fetch_upstream_license(upstream["hf_repo"], revision_pin, staging)
    except ImportError:
        err('huggingface_hub is not installed. Install: pip install "coreai-fabric[hf]"')
        return 1
    if not license_files and not args.allow_missing_license_file:
        err(f"could not mirror an upstream LICENSE/NOTICE from {upstream['hf_repo']} — "
            "refusing to redistribute weights without their license text "
            "(Apache-2.0 §4(a)). If the upstream genuinely ships none, re-run with "
            "--allow-missing-license-file.")
        return 1
    holder = copyright_holder_from_license(staging)

    # Multi-variant repo (community layout): a `publish.variant` tier lands its
    # bundle + reports under `<variant>/`, while the shared card + LICENSE sit at
    # the repo root and the card compares all tiers.
    variant = publish_cfg.get("variant")
    bundle_repo_path = f"{variant}/{bundle.name}" if variant else bundle.name
    card = render_model_card(root, recipe, manifest, report, copyright_holder=holder)
    (staging / "README.md").write_text(card)
    report_dir = (staging / variant) if variant else staging
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "parity-report.json").write_text(json.dumps(report, indent=2) + "\n")
    if manifest:
        (report_dir / "reproduce-manifest.json").write_text(
            json.dumps(sanitize_manifest(manifest, root), indent=2) + "\n")

    # Last-line privacy guard: no absolute local path may reach a public repo.
    leaks = assert_no_local_paths(staging)
    if leaks:
        err(f"ABORT: local paths would leak into the public repo via {leaks} — "
            "not uploading. This is a bug; report it.")
        return 1

    if args.dry_run:
        print(f"would publish to https://huggingface.co/{repo_id}")
        staged = sorted(str(p.relative_to(staging)) for p in staging.rglob("*") if p.is_file())
        print(f"  fileset: {bundle_repo_path}/ + " + ", ".join(staged))
        print("--- model card (repo root README.md) ---")
        print(card)
        return 0

    from huggingface_hub import HfApi
    api = HfApi()

    # Atomic-ish publish: create PRIVATE, land weights + card + license, then
    # flip public as the final step. A failure mid-upload never leaves a PUBLIC
    # license-less, un-attributed weights drop (Apache-2.0 §4 + reputation).
    print(f"creating repo (private): {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
    _set_private(api, repo_id, True)
    print(f"uploading bundle -> {bundle_repo_path} + card + license (private) ...")
    api.upload_folder(repo_id=repo_id, folder_path=str(bundle), path_in_repo=bundle_repo_path,
                      commit_message=f"coreai-fabric: {recipe.id} asset")
    info = api.upload_folder(repo_id=repo_id, folder_path=str(staging), path_in_repo=".",
                             commit_message=f"coreai-fabric: {recipe.id} card + license + reports")
    revision = getattr(info, "oid", None) or api.model_info(repo_id).sha

    collection_title = publish_cfg.get("collection")
    collection_url = None
    if collection_title:
        collection_url = _add_to_collection(
            api, publish_cfg["hf_target_namespace"], collection_title, repo_id)
        if collection_url:
            # Re-render the card now that we know the collection URL, and update it.
            card = render_model_card(root, recipe, manifest, report,
                                     copyright_holder=holder, collection_url=collection_url)
            api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                            repo_id=repo_id, commit_message="coreai-fabric: link collection")

    # Everything landed — make it public.
    _set_private(api, repo_id, False)
    ok(f"published https://huggingface.co/{repo_id} @ {revision}")
    if collection_url:
        ok(f"added to collection '{collection_title}': {collection_url}")

    recipe.data["published"] = {"hf_repo": repo_id, "revision": revision, "date": today()}
    recipe.data["status"] = "published"
    write_yaml(recipe.path, recipe.data)
    ok(f"recipe status -> published ({recipe.path.name})")
    print(f"next: coreai-fabric register {recipe.id} --catalog-path ../coreai-catalog --dry-run")
    return 0


def _set_private(api, repo_id: str, private: bool) -> None:
    """Flip repo visibility, tolerant of huggingface_hub API differences."""
    for attempt in (
        lambda: api.update_repo_settings(repo_id=repo_id, private=private),
        lambda: api.update_repo_visibility(repo_id=repo_id, private=private),
    ):
        try:
            attempt()
            return
        except (AttributeError, TypeError):
            continue
        except Exception as exc:  # noqa: BLE001
            warn(f"could not set repo private={private}: {exc}")
            return
