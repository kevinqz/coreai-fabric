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


def fetch_upstream_license(hf_repo: str, revision: str | None, staging: Path,
                           root: Path | None = None,
                           declared_license: str | None = None) -> list[str]:
    """Download the upstream LICENSE/NOTICE into staging (Apache-2.0 §4(a)/(d):
    a redistribution must ship a copy of the license + retain the NOTICE). The
    caller treats an absent-but-required license as a hard publish failure —
    fabric never redistributes weights without their license text.

    Fallback: if the upstream ships NO license file but its model-card metadata
    authoritatively DECLARES a known SPDX license (common for lerobot policies,
    which tag `license:apache-2.0` yet ship no LICENSE), supply the canonical
    license text + a provenance NOTICE. That gives recipients the full terms as
    §4(a) requires — strictly more compliant than shipping license-less weights,
    and only ever for a license fabric has canonical text for."""
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
    if written:
        return written

    # No upstream license file — synthesize the canonical text iff the upstream
    # DECLARES a license we have verbatim text for (never invent terms).
    if root is not None and declared_license:
        spdx = str(declared_license).strip().lower()
        tmpl = root / "templates" / "licenses" / f"{spdx}.txt"
        if tmpl.is_file():
            rev = revision or "main"
            if spdx == "gemma":
                # The Gemma Terms of Use §3.1 impose MORE than shipping the license
                # text: a redistribution must also (a) carry a NOTICE file with a
                # verbatim mandated string, (b) give prominent notice that the files
                # were modified, and (c) pass through the §3.2 use restrictions +
                # Prohibited Use Policy. The pi0/pi0.5 policies embed PaliGemma/Gemma
                # weights, so the CoreAI asset is a Gemma "Model Derivative" and all
                # three obligations apply. Never redistribute Gemma weights without them.
                header = (
                    "NOTE — supplied by coreai-fabric (the redistributor), not the upstream.\n"
                    f'The upstream {hf_repo} @ {rev} declares its license as "gemma" in its\n'
                    "Hugging Face model-card metadata but ships no LICENSE file. These weights\n"
                    'are a Gemma "Model Derivative": they embed PaliGemma/Gemma parameters via\n'
                    "the LeRobot pi0/pi0.5 policy. Under the Gemma Terms of Use §3.1 a\n"
                    "redistribution must provide recipients a copy of the Agreement, so the\n"
                    "canonical Gemma Terms of Use are reproduced below verbatim. Copyright in\n"
                    f"the underlying work remains with Google and the upstream authors ({hf_repo}).\n"
                    + "=" * 72 + "\n\n"
                )
                (staging / "LICENSE").write_text(header + tmpl.read_text())
                # §3.1(d): the NOTICE must CONTAIN this exact string. It MUST stay on a
                # single physical line — a wrapped newline breaks the verbatim substring
                # (an adversarial audit caught this). Do not reflow this line.
                MANDATED = ("Gemma is provided under and subject to the Gemma Terms of Use "
                            "found at ai.google.dev/gemma/terms")
                notice = (
                    MANDATED + "\n\n"
                    "MODIFICATION NOTICE (Gemma Terms of Use §3.1): the model files in this\n"
                    "repository are a Model Derivative of Gemma. The upstream LeRobot pi0/pi0.5\n"
                    "policy embeds PaliGemma/Gemma trained weights; coreai-fabric has further\n"
                    "modified them by converting the policy to Apple's Core AI on-device format\n"
                    f"(.aimodel, float16, coreai-optimized). Upstream: {hf_repo} @ {rev}.\n\n"
                    "USE RESTRICTIONS (Gemma Terms of Use §3.2): use of these files is subject to\n"
                    "the Gemma Prohibited Use Policy at ai.google.dev/gemma/prohibited_use_policy,\n"
                    "incorporated by reference into the Agreement, and must not violate applicable\n"
                    "law.\n\n"
                    "FLOW-DOWN (Gemma Terms of Use §3.1): if you further Distribute these files or\n"
                    "any Model Derivative of them, you must (a) provide recipients a copy of the\n"
                    "Gemma Terms of Use (the bundled LICENSE), (b) reproduce this NOTICE including\n"
                    "the mandated string above, (c) carry the §3.2 use restrictions as an\n"
                    "enforceable provision in your governing agreement, and (d) prominently state\n"
                    "any further modifications. Copyright in the underlying Gemma weights remains\n"
                    "with Google LLC; see the model card and LICENSE for attribution.\n"
                )
                (staging / "NOTICE").write_text(notice)
                return ["LICENSE", "NOTICE"]
            header = (
                "NOTE — supplied by coreai-fabric (the redistributor), not the upstream.\n"
                f"The upstream {hf_repo} @ {rev} declares its license as "
                f'"{declared_license}"\n'
                "in its Hugging Face model-card metadata but ships no LICENSE file. Under\n"
                f"Apache-2.0 §4(a) a redistribution must give recipients a copy of the\n"
                f"License, so the canonical {declared_license} text is reproduced below\n"
                "verbatim. Copyright in the underlying work remains with the upstream\n"
                f"authors ({hf_repo}); see the model card for attribution.\n"
                + "=" * 72 + "\n\n"
            )
            (staging / "LICENSE").write_text(header + tmpl.read_text())
            return ["LICENSE"]
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


def _machine_chip_brand() -> str | None:
    """The publisher's specific CPU brand (e.g. 'Apple Silicon') — a hardware fingerprint that must
    never ship in a repo. Used only as a forbidden marker for the pre-upload guard."""
    import subprocess
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except OSError:
        return None


def assert_no_local_paths(staging: Path) -> list[str]:
    """Last-line guard before a public upload: any staged text file still carrying an absolute local
    path OR the publisher's specific hardware fingerprint (chip brand string) is a leak. Returns the
    offending filenames."""
    chip = _machine_chip_brand()
    markers = (*_LOCAL_PATH_MARKERS, str(Path.home()), *([chip] if chip else []))
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
    """The variant tiers ACTUALLY present in the repo (namespace+repo_name):
    the tier being published now (the current recipe) plus any already
    published. A recipe that merely *targets* the repo but hasn't been
    published is excluded — otherwise the card would advertise a `<variant>/`
    subdir that 404s (e.g. a drafted int4 next to a just-published int8).
    Returns [(variant, recipe, parity_report_or_None)] sorted by variant name."""
    from .recipes import load_all_recipes
    pub = recipe.data.get("publish", {})
    key = (pub.get("hf_target_namespace"), pub.get("repo_name"))
    out = []
    for r in load_all_recipes(root):
        rp = r.data.get("publish", {})
        if not rp.get("variant") or (rp.get("hf_target_namespace"), rp.get("repo_name")) != key:
            continue
        # In the repo iff it's the current publish or was already published.
        if r.id != recipe.id and not r.data.get("published"):
            continue
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

    # S2: the rich card template is LLM-specific — a chat hook, a "stateful
    # KV-cache chat" asset description, and a CoreAILanguageModel Swift example.
    # Rendering it for a non-LLM bundle would MISLABEL the asset (a whisper .aimodel
    # is not a chat model). Fail loud with the fix rather than ship a lying card:
    # author a per-kind template (honest hook + the io_contract-driven Swift
    # snippet) and dispatch on bundle_kind. Best done alongside the first real
    # conversion of that kind, so the card is validated against a real asset.
    bundle_kind = catalog_block.get("bundle_kind")
    if bundle_kind == "action":
        # A robot policy gets the honest action card (needs-a-robot banner,
        # action_parity, NO chat language) — never the LLM template.
        return _render_action_card(root, recipe, manifest, report,
                                   copyright_holder=copyright_holder,
                                   collection_url=collection_url)
    if bundle_kind == "token-classification":
        # A non-AR encoder gets the honest token-classification card (encoder, not
        # chat; host owns the tokenizer; Gate B is graph_output_cosine).
        return _render_token_classification_card(root, recipe, manifest, report,
                                                 copyright_holder=copyright_holder,
                                                 collection_url=collection_url)
    if bundle_kind == "image-feature-extraction":
        # A frozen vision backbone gets the honest feature-extraction card (image
        # -> per-patch tokens; host owns preprocessing; Gate B graph_output_cosine).
        return _render_image_feature_extraction_card(root, recipe, manifest, report,
                                                     copyright_holder=copyright_holder,
                                                     collection_url=collection_url)
    if bundle_kind == "reward-model":
        # A reward model's deployable core is small MLP heads over host-owned VLM
        # hidden states — honest reward-head card (NO chat/task-success language;
        # Gate B graph_output_cosine).
        return _render_reward_model_card(root, recipe, manifest, report,
                                         copyright_holder=copyright_holder,
                                         collection_url=collection_url)
    if bundle_kind and bundle_kind != "llm":
        raise SystemExit(
            f"publish: no card template for bundle_kind '{bundle_kind}' yet "
            f"(only llm + action). Publishing {recipe.id} now would mislabel a "
            f"{bundle_kind} asset. Author templates/model-card-{bundle_kind}.md + a "
            f"dispatch branch in render_model_card first — fabric will not ship a "
            f"card that misdescribes the asset."
        )

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


def _render_action_card(root: Path, recipe, manifest: dict, report: dict,
                        *, copyright_holder: str | None = None,
                        collection_url: str | None = None) -> str:
    """Card for a VLA/robot policy (bundle_kind: action). Honest by construction:
    a needs-a-robot banner, action_parity verification (or an explicit not_run),
    and NO chat language — a policy is not a chat model."""
    from . import FABRIC_REPO_URL
    upstream = recipe.data["upstream"]
    conv = recipe.data["conversion"]
    catalog_block = recipe.data.get("catalog") or {}
    publish_cfg = recipe.data.get("publish") or {}
    action = conv.get("action") or {}
    repo_id = f"{publish_cfg.get('hf_target_namespace')}/{publish_cfg.get('repo_name')}"
    template = (root / "templates" / "model-card-action.md").read_text()

    quant = str(conv.get("quantization", "none")).strip()
    is_quantized = quant.lower() not in ("", "none")
    base_model_relation_line = "base_model_relation: quantized\n" if is_quantized else ""

    tags = ["coreai", "core-ai", "coreai-fabric", "aimodel", "coreml", "apple",
            "apple-silicon", "on-device", "robotics", "vla",
            "vision-language-action", "action"]
    if is_quantized:
        tags.append(quant.lower())
    seen: set[str] = set()
    tags_block = "".join(f"- {t}\n" for t in tags if not (t in seen or seen.add(t)))

    langs = catalog_block.get("languages") or upstream.get("languages")
    language_block = ("language:\n" + "".join(f"- {l}\n" for l in langs)) if langs else ""

    # Embodiment + sampling — from the recipe, never invented.
    embodiment = (action.get("embodiment") or catalog_block.get("embodiment")
                  or "the robot embodiment it was trained on (see the upstream card)")
    sampling = ((action.get("sampling") or {}).get("kind")
                or catalog_block.get("sampling") or "vision-language-action")
    steps = (action.get("sampling") or {}).get("num_steps")
    _graphs = action.get("graphs") or []

    meta = _bundle_metadata(root, recipe)
    min_os = catalog_block.get("min_os") or {}
    min_os_str = f"macOS {min_os.get('macos', '27.0')}+ / iOS {min_os.get('ios', '27.0')}+"
    main = bundle_path(root, recipe) / "main.mlirb"
    size_str = _human_size(main.stat().st_size) if main.is_file() else "—"
    facts_rows = [
        ("Parameters", catalog_block.get("parameters", "—")),
        ("Architecture", catalog_block.get("architecture", "—")),
        ("Capabilities", ", ".join(catalog_block.get("capabilities", [])) or "—"),
        ("Embodiment", embodiment),
        ("Sampling", f"{sampling}" + (f" ({steps}-step)" if steps else "")),
        ("Quantization / precision", f"{conv.get('quantization', '—')} / {conv.get('precision', '—')}"),
        ("On-disk size", size_str),
        ("Asset kind", (
            f"split-export policy ({', '.join(g.get('name', '?') for g in _graphs)}) + norm_stats"
            if len(_graphs) > 1 else "single-graph policy + norm_stats sidecar")),
        ("assetVersion", meta.get("assetVersion", "—")),
    ]
    facts_block = "".join(f"| {k} | {v} |\n" for k, v in facts_rows)

    gb = report.get("gate_b", {})
    if gb.get("metric") == "action_parity" and isinstance(gb.get("value"), (int, float)):
        env = gb.get("environment", {})
        dev = env.get("chip") or "Apple Silicon"
        ref = {"float16": "fp16", "float32": "fp32"}.get(gb.get("reference_dtype"), gb.get("reference_dtype", "fp16"))
        cos = round(100 * gb["value"], 1)
        mae = gb.get("max_normalized_mae")
        n = gb.get("compared") or gb.get("n_obs") or "N"
        ds = gb.get("dataset", "recorded episodes")
        evaluation_block = (
            f"- **Gate B — action_parity: {cos}% min chunk-cosine"
            + (f" · {round(mae, 4)} max normalized MAE" if isinstance(mae, (int, float)) else "")
            + f"** vs the {ref} reference over {n} frames of `{ds}`, "
            f"{gb.get('num_steps') or 'N'}-step, fixed-noise (measured on {dev})."
        )
        # Full transparency for near-zero-action-conditioned passes: chunk-cosine is ill-conditioned
        # where the reference action is near zero (the robot barely moves), so the strict MIN can dip
        # while the export stays behaviorally faithful. Surface the median + absolute error + caveat
        # PROMINENTLY on the card — never hide the raw min.
        if gb.get("near_zero_conditioned"):
            med = gb.get("median_action_cosine")
            amae = gb.get("max_per_dim_mae")
            evaluation_block += (
                f"\n- **Near-zero-action note (full disclosure):** the {cos}% is the raw MIN over "
                f"all frames, dominated by frame(s) where the reference action is ~zero (the robot "
                f"barely moves) and chunk-cosine is mathematically ill-conditioned. The **median "
                f"chunk-cosine is {round(100 * med, 1) if isinstance(med, (int, float)) else '—'}%** "
                f"and the **max ABSOLUTE per-dim action error is "
                f"{round(amae, 4) if isinstance(amae, (int, float)) else '—'}** (normalized) — i.e. "
                f"behaviorally faithful everywhere. This model class is gated on the absolute action "
                f"error (behaviorally meaningful) since cosine is undefined for ~zero actions; every "
                f"raw number is reported here and in `parity-report.json`. fabric never fakes a number."
            )
    else:
        evaluation_block = (
            f"- **Gate B (action_parity): {gb.get('status', 'not_run')}** — "
            "unmeasured pending a real conversion + `verify` on hardware. "
            "fabric never fakes a parity number."
        )

    # Gemma redistribution: the Gemma Terms give NO 'Copyright <year>' line, so
    # copyright_holder_from_license() returns None. Name Google explicitly (an
    # adversarial audit flagged a holder-less card as an attribution gap), surface
    # the §3.2 restrictions + Prohibited Use Policy in the BODY (not only NOTICE),
    # and add the §4.2 non-endorsement disclaimer for Google.
    is_gemma = str(upstream.get("license", "")).strip().lower() == "gemma"
    holder = copyright_holder or upstream.get("copyright_holder")
    if is_gemma and not holder:
        holder = "Google LLC and the upstream authors"
    attribution = (f"Weights © {holder}, " if holder else "Weights ") + \
        f"licensed **{upstream['license']}** — see the bundled `LICENSE`."
    if is_gemma:
        gemma_license_block = (
            "\n\nThese weights are a **Model Derivative of Gemma** — the upstream pi0/pi0.5 "
            "policy embeds PaliGemma/Gemma parameters. Redistribution and use are governed by the "
            "[Gemma Terms of Use](https://ai.google.dev/gemma/terms) (bundled as `LICENSE`) and "
            "restricted by the [Gemma Prohibited Use Policy]"
            "(https://ai.google.dev/gemma/prohibited_use_policy) (§3.2, incorporated by reference). "
            "Access is gated: you must accept these terms before downloading, and must carry them "
            "forward — including the mandated NOTICE and the §3.2 restrictions — on any further "
            "distribution (see [`NOTICE`](./NOTICE)). **Gemma** and **PaliGemma** are trademarks of "
            "Google LLC, used here only descriptively; this conversion is **not affiliated with, "
            "produced by, or endorsed by Google**."
        )
        gated_frontmatter = (
            'extra_gated_heading: "Access this Gemma Model Derivative"\n'
            "extra_gated_prompt: >-\n"
            "  These weights are a Model Derivative of Gemma. Access and use are governed by the\n"
            "  Gemma Terms of Use (https://ai.google.dev/gemma/terms) and the Gemma Prohibited\n"
            "  Use Policy (https://ai.google.dev/gemma/prohibited_use_policy). By requesting access\n"
            "  you agree to those terms and the Section 3.2 use restrictions, and you agree to carry\n"
            "  these terms forward on any further distribution.\n"
            "extra_gated_fields:\n"
            "  I agree to the Gemma Terms of Use and the Gemma Prohibited Use Policy: checkbox\n"
            "  I agree to carry these terms forward on any redistribution: checkbox\n"
        )
    else:
        gemma_license_block = ""
        gated_frontmatter = ""

    mirror_ns = publish_cfg.get("mirror_namespace")
    mirror_line = (
        f"> **Canonical:** [`{repo_id}`](https://huggingface.co/{repo_id}) — source of truth. "
        + (f"**Mirror:** [`{mirror_ns}/{publish_cfg.get('repo_name')}`](https://huggingface.co/{mirror_ns}/{publish_cfg.get('repo_name')})."
           if mirror_ns else ""))
    collection_link = f"- [HF Collection]({collection_url})\n" if collection_url else ""

    return template.format(
        license=upstream["license"],
        upstream_hf_repo=upstream["hf_repo"],
        upstream_revision=manifest.get("input", {}).get("revision") or upstream.get("revision", "unpinned"),
        base_model_relation_line=base_model_relation_line,
        gated_frontmatter=gated_frontmatter,
        gemma_license_block=gemma_license_block,
        language_block=language_block,
        tags_block=tags_block,
        name=catalog_block.get("name", recipe.id),
        embodiment=embodiment,
        sampling=sampling,
        recipe_id=recipe.id,
        mirror_line=mirror_line,
        facts_block=facts_block,
        min_os=min_os_str,
        gate_a_status=report["gate_a"]["status"],
        evaluation_block=evaluation_block,
        attribution=attribution,
        collection_link=collection_link,
        recipe_url=f"{FABRIC_REPO_URL}/blob/main/recipes/{recipe.id}.yaml",
        tool=manifest.get("tool", conv["tool"]),
        tool_version=manifest.get("tool_version") or "(version not reported)",
        precision=conv["precision"],
        quantization=conv["quantization"],
        date=today(),
    )


def _render_token_classification_card(root: Path, recipe, manifest: dict, report: dict,
                                      *, copyright_holder: str | None = None,
                                      collection_url: str | None = None) -> str:
    """Card for a non-autoregressive encoder (bundle_kind: token-classification).
    Honest by construction: an encoder that labels tokens (NO chat language), the
    host owns the tokenizer, and Gate B is graph_output_cosine (output parity)."""
    from . import FABRIC_REPO_URL
    upstream = recipe.data["upstream"]
    conv = recipe.data["conversion"]
    catalog_block = recipe.data.get("catalog") or {}
    publish_cfg = recipe.data.get("publish") or {}
    repo_id = f"{publish_cfg.get('hf_target_namespace')}/{publish_cfg.get('repo_name')}"
    template = (root / "templates" / "model-card-token-classification.md").read_text()

    quant = str(conv.get("quantization", "none")).strip()
    is_quantized = quant.lower() not in ("", "none")
    base_model_relation_line = "base_model_relation: quantized\n" if is_quantized else ""

    tags = ["coreai", "core-ai", "coreai-fabric", "aimodel", "coreml", "apple",
            "apple-silicon", "on-device", "encoder", "token-classification"]
    if is_quantized:
        tags.append(quant.lower())
    seen: set[str] = set()
    tags_block = "".join(f"- {t}\n" for t in tags if not (t in seen or seen.add(t)))

    langs = catalog_block.get("languages") or upstream.get("languages")
    language_block = ("language:\n" + "".join(f"- {l}\n" for l in langs)) if langs else ""

    # Contract (num_labels, seq_len) from the export — never invented.
    contract: dict = {}
    cpath = bundle_path(root, recipe).parent / "eurobert-contract.json"
    if cpath.is_file():
        try:
            contract = json.loads(cpath.read_text())
        except Exception:  # noqa: BLE001
            contract = {}

    meta = _bundle_metadata(root, recipe)
    min_os = catalog_block.get("min_os") or {}
    min_os_str = f"macOS {min_os.get('macos', '27.0')}+ / iOS {min_os.get('ios', '27.0')}+"
    main = bundle_path(root, recipe) / "main.mlirb"
    size_str = _human_size(main.stat().st_size) if main.is_file() else "—"
    facts_rows = [
        ("Parameters", catalog_block.get("parameters", "—")),
        ("Architecture", catalog_block.get("architecture", "—")),
        ("Capabilities", ", ".join(catalog_block.get("capabilities", [])) or "—"),
        ("Labels", contract.get("num_labels", "—")),
        ("Sequence length", f"{contract.get('seq_len', '—')} (static)"),
        ("Quantization / precision", f"{conv.get('quantization', '—')} / {conv.get('precision', '—')}"),
        ("On-disk size", size_str),
        ("Asset kind", "single-graph encoder ((input_ids, attention_mask) -> per-token logits)"),
        ("assetVersion", meta.get("assetVersion", "—")),
    ]
    facts_block = "".join(f"| {k} | {v} |\n" for k, v in facts_rows)

    gb = report.get("gate_b", {})
    if gb.get("metric") == "graph_output_cosine" and isinstance(gb.get("value"), (int, float)):
        env = gb.get("environment", {})
        dev = env.get("accelerator") or env.get("chip") or "Apple Silicon"
        ref = {"float16": "fp16", "float32": "fp32"}.get(
            gb.get("reference_dtype"), gb.get("reference_dtype", "fp32"))
        med = gb.get("median_cosine")
        med_str = f" (median {med:.6f})" if isinstance(med, (int, float)) else ""
        n = gb.get("n_obs") or "N"
        evaluation_block = (
            f"- **Gate B — graph_output_cosine: {gb['value']:.6f} min output cosine**{med_str} "
            f"vs the {ref} torch reference over {n} seeded `(input_ids, attention_mask)`, measured "
            f"on {dev}. The encoder analog of the LLM logit-parity: it certifies the export computes "
            f"the SAME per-token logits as the source — a conversion-fidelity metric, not task accuracy."
        )
    else:
        evaluation_block = (
            f"- **Gate B (graph_output_cosine): {gb.get('status', 'not_run')}** — unmeasured "
            "pending a real conversion + `verify` on hardware.")

    holder = copyright_holder or upstream.get("copyright_holder")
    attribution = (f"Weights © {holder}, " if holder else "Weights ") + \
        f"licensed **{upstream['license']}** — see the bundled `LICENSE`."

    mirror_ns = publish_cfg.get("mirror_namespace")
    mirror_line = (
        f"> **Canonical:** [`{repo_id}`](https://huggingface.co/{repo_id}) — source of truth. "
        + (f"**Mirror:** [`{mirror_ns}/{publish_cfg.get('repo_name')}`]"
           f"(https://huggingface.co/{mirror_ns}/{publish_cfg.get('repo_name')})."
           if mirror_ns else ""))
    collection_link = f"- [HF Collection]({collection_url})\n" if collection_url else ""

    return template.format(
        license=upstream["license"],
        upstream_hf_repo=upstream["hf_repo"],
        upstream_revision=manifest.get("input", {}).get("revision") or upstream.get("revision", "unpinned"),
        base_model_relation_line=base_model_relation_line,
        gated_frontmatter="",
        gemma_license_block="",
        language_block=language_block,
        tags_block=tags_block,
        name=catalog_block.get("name", recipe.id),
        recipe_id=recipe.id,
        mirror_line=mirror_line,
        facts_block=facts_block,
        min_os=min_os_str,
        gate_a_status=report["gate_a"]["status"],
        evaluation_block=evaluation_block,
        attribution=attribution,
        collection_link=collection_link,
        recipe_url=f"{FABRIC_REPO_URL}/blob/main/recipes/{recipe.id}.yaml",
        tool=manifest.get("tool", conv["tool"]),
        tool_version=manifest.get("tool_version") or "(version not reported)",
        precision=conv["precision"],
        quantization=conv["quantization"],
        date=today(),
    )


def _render_image_feature_extraction_card(root: Path, recipe, manifest: dict, report: dict,
                                          *, copyright_holder: str | None = None,
                                          collection_url: str | None = None) -> str:
    """Card for a frozen vision backbone (bundle_kind: image-feature-extraction).
    Honest by construction: an image encoder that emits per-patch feature tokens
    (NO end task), host owns preprocessing, Gate B is graph_output_cosine."""
    from . import FABRIC_REPO_URL
    upstream = recipe.data["upstream"]
    conv = recipe.data["conversion"]
    catalog_block = recipe.data.get("catalog") or {}
    publish_cfg = recipe.data.get("publish") or {}
    repo_id = f"{publish_cfg.get('hf_target_namespace')}/{publish_cfg.get('repo_name')}"
    template = (root / "templates" / "model-card-image-feature-extraction.md").read_text()

    quant = str(conv.get("quantization", "none")).strip()
    is_quantized = quant.lower() not in ("", "none")
    base_model_relation_line = "base_model_relation: quantized\n" if is_quantized else ""

    tags = ["coreai", "core-ai", "coreai-fabric", "aimodel", "coreml", "apple",
            "apple-silicon", "on-device", "vision", "image-feature-extraction", "vit"]
    if is_quantized:
        tags.append(quant.lower())
    seen: set[str] = set()
    tags_block = "".join(f"- {t}\n" for t in tags if not (t in seen or seen.add(t)))

    langs = catalog_block.get("languages") or upstream.get("languages")
    language_block = ("language:\n" + "".join(f"- {l}\n" for l in langs)) if langs else ""

    contract: dict = {}
    cpath = bundle_path(root, recipe).parent / "lingbot-contract.json"
    if cpath.is_file():
        try:
            contract = json.loads(cpath.read_text())
        except Exception:  # noqa: BLE001
            contract = {}

    meta = _bundle_metadata(root, recipe)
    min_os = catalog_block.get("min_os") or {}
    min_os_str = f"macOS {min_os.get('macos', '27.0')}+ / iOS {min_os.get('ios', '27.0')}+"
    main = bundle_path(root, recipe) / "main.mlirb"
    size_str = _human_size(main.stat().st_size) if main.is_file() else "—"
    facts_rows = [
        ("Parameters", catalog_block.get("parameters", "—")),
        ("Architecture", catalog_block.get("architecture", "—")),
        ("Capabilities", ", ".join(catalog_block.get("capabilities", [])) or "—"),
        ("Image size", f"{contract.get('image_size', '—')}px (static)"),
        ("Patch size", contract.get("patch_size", "—")),
        ("Embed dim", contract.get("embed_dim", "—")),
        ("Patch tokens", contract.get("num_patch_tokens", "—")),
        ("Quantization / precision", f"{conv.get('quantization', '—')} / {conv.get('precision', '—')}"),
        ("On-disk size", size_str),
        ("Asset kind", "single-graph ViT encoder (image -> per-patch tokens)"),
        ("assetVersion", meta.get("assetVersion", "—")),
    ]
    facts_block = "".join(f"| {k} | {v} |\n" for k, v in facts_rows)

    gb = report.get("gate_b", {})
    if gb.get("metric") == "graph_output_cosine" and isinstance(gb.get("value"), (int, float)):
        env = gb.get("environment", {})
        dev = env.get("accelerator") or env.get("chip") or "Apple Silicon"
        ref = {"float16": "fp16", "float32": "fp32"}.get(
            gb.get("reference_dtype"), gb.get("reference_dtype", "fp32"))
        med = gb.get("median_cosine")
        med_str = f" (median {med:.6f})" if isinstance(med, (int, float)) else ""
        n = gb.get("n_obs") or "N"
        evaluation_block = (
            f"- **Gate B — graph_output_cosine: {gb['value']:.6f} min output cosine**{med_str} "
            f"vs the {ref} torch backbone over {n} seeded images, measured on {dev}. Certifies the "
            f"export computes the SAME per-patch tokens as the source backbone — a conversion-fidelity "
            f"metric, not task accuracy."
        )
    else:
        evaluation_block = (
            f"- **Gate B (graph_output_cosine): {gb.get('status', 'not_run')}** — unmeasured "
            "pending a real conversion + `verify` on hardware.")

    holder = copyright_holder or upstream.get("copyright_holder")
    attribution = (f"Weights © {holder}, " if holder else "Weights ") + \
        f"licensed **{upstream['license']}** — see the bundled `LICENSE`."

    mirror_ns = publish_cfg.get("mirror_namespace")
    mirror_line = (
        f"> **Canonical:** [`{repo_id}`](https://huggingface.co/{repo_id}) — source of truth. "
        + (f"**Mirror:** [`{mirror_ns}/{publish_cfg.get('repo_name')}`]"
           f"(https://huggingface.co/{mirror_ns}/{publish_cfg.get('repo_name')})."
           if mirror_ns else ""))
    collection_link = f"- [HF Collection]({collection_url})\n" if collection_url else ""

    return template.format(
        license=upstream["license"],
        upstream_hf_repo=upstream["hf_repo"],
        upstream_revision=manifest.get("input", {}).get("revision") or upstream.get("revision", "unpinned"),
        base_model_relation_line=base_model_relation_line,
        gated_frontmatter="",
        gemma_license_block="",
        language_block=language_block,
        tags_block=tags_block,
        name=catalog_block.get("name", recipe.id),
        recipe_id=recipe.id,
        mirror_line=mirror_line,
        facts_block=facts_block,
        min_os=min_os_str,
        gate_a_status=report["gate_a"]["status"],
        evaluation_block=evaluation_block,
        attribution=attribution,
        collection_link=collection_link,
        recipe_url=f"{FABRIC_REPO_URL}/blob/main/recipes/{recipe.id}.yaml",
        tool=manifest.get("tool", conv["tool"]),
        tool_version=manifest.get("tool_version") or "(version not reported)",
        precision=conv["precision"],
        quantization=conv["quantization"],
        date=today(),
    )


def _render_reward_model_card(root: Path, recipe, manifest: dict, report: dict,
                              *, copyright_holder: str | None = None,
                              collection_url: str | None = None) -> str:
    """Card for a robot-policy reward head (bundle_kind: reward-model).
    Honest by construction: ships ONLY the MLP reward heads (progress + success)
    over host-owned VLM hidden states — NO task-success claim; Gate B is
    graph_output_cosine."""
    from . import FABRIC_REPO_URL
    upstream = recipe.data["upstream"]
    conv = recipe.data["conversion"]
    catalog_block = recipe.data.get("catalog") or {}
    publish_cfg = recipe.data.get("publish") or {}
    repo_id = f"{publish_cfg.get('hf_target_namespace')}/{publish_cfg.get('repo_name')}"
    template = (root / "templates" / "model-card-reward-model.md").read_text()

    quant = str(conv.get("quantization", "none")).strip()
    is_quantized = quant.lower() not in ("", "none")
    base_model_relation_line = "base_model_relation: quantized\n" if is_quantized else ""

    tags = ["coreai", "core-ai", "coreai-fabric", "aimodel", "coreml", "apple",
            "apple-silicon", "on-device", "robotics", "reward-model", "lerobot"]
    if is_quantized:
        tags.append(quant.lower())
    seen: set[str] = set()
    tags_block = "".join(f"- {t}\n" for t in tags if not (t in seen or seen.add(t)))

    langs = catalog_block.get("languages") or upstream.get("languages")
    language_block = ("language:\n" + "".join(f"- {l}\n" for l in langs)) if langs else ""

    contract: dict = {}
    cpath = bundle_path(root, recipe).parent / "robometer-reward-contract.json"
    if cpath.is_file():
        try:
            contract = json.loads(cpath.read_text())
        except Exception:  # noqa: BLE001
            contract = {}

    meta = _bundle_metadata(root, recipe)
    min_os = catalog_block.get("min_os") or {}
    min_os_str = f"macOS {min_os.get('macos', '27.0')}+ / iOS {min_os.get('ios', '27.0')}+"
    main = bundle_path(root, recipe) / "main.mlirb"
    size_str = _human_size(main.stat().st_size) if main.is_file() else "—"
    facts_rows = [
        ("Parameters (full model)", catalog_block.get("parameters", "—")),
        ("Architecture", catalog_block.get("architecture", "—")),
        ("Capabilities", ", ".join(catalog_block.get("capabilities", [])) or "—"),
        ("Hidden dim (VLM)", contract.get("hidden_dim", "—")),
        ("Progress bins", contract.get("progress_bins", "—")),
        ("Max frames (static)", contract.get("max_frames", "—")),
        ("Outputs", ", ".join(contract.get("outputs", [])) or "—"),
        ("Quantization / precision", f"{conv.get('quantization', '—')} / {conv.get('precision', '—')}"),
        ("On-disk size", size_str),
        ("Asset kind", "MLP reward heads (VLM hidden states -> progress + success logits)"),
        ("assetVersion", meta.get("assetVersion", "—")),
    ]
    facts_block = "".join(f"| {k} | {v} |\n" for k, v in facts_rows)

    gb = report.get("gate_b", {})
    if gb.get("metric") == "graph_output_cosine" and isinstance(gb.get("value"), (int, float)):
        env = gb.get("environment", {})
        dev = env.get("accelerator") or env.get("chip") or "Apple Silicon"
        ref = {"float16": "fp16", "float32": "fp32"}.get(
            gb.get("reference_dtype"), gb.get("reference_dtype", "fp32"))
        med = gb.get("median_cosine")
        med_str = f" (median {med:.6f})" if isinstance(med, (int, float)) else ""
        n = gb.get("n_obs") or "N"
        evaluation_block = (
            f"- **Gate B — graph_output_cosine: {gb['value']:.6f} min output cosine**{med_str} "
            f"vs the {ref} torch reward heads over {n} seeded hidden-state inputs (worst of the "
            f"progress + success heads), measured on {dev}. Certifies the export computes the SAME "
            f"reward-head logits as the source — a conversion-fidelity metric, not reward quality."
        )
    else:
        evaluation_block = (
            f"- **Gate B (graph_output_cosine): {gb.get('status', 'not_run')}** — unmeasured "
            "pending a real conversion + `verify` on hardware.")

    holder = copyright_holder or upstream.get("copyright_holder")
    attribution = (f"Weights © {holder}, " if holder else "Weights ") + \
        f"licensed **{upstream['license']}** — see the bundled `LICENSE`."

    mirror_ns = publish_cfg.get("mirror_namespace")
    mirror_line = (
        f"> **Canonical:** [`{repo_id}`](https://huggingface.co/{repo_id}) — source of truth. "
        + (f"**Mirror:** [`{mirror_ns}/{publish_cfg.get('repo_name')}`]"
           f"(https://huggingface.co/{mirror_ns}/{publish_cfg.get('repo_name')})."
           if mirror_ns else ""))
    collection_link = f"- [HF Collection]({collection_url})\n" if collection_url else ""

    return template.format(
        license=upstream["license"],
        upstream_hf_repo=upstream["hf_repo"],
        upstream_revision=manifest.get("input", {}).get("revision") or upstream.get("revision", "unpinned"),
        base_model_relation_line=base_model_relation_line,
        gated_frontmatter="",
        gemma_license_block="",
        language_block=language_block,
        tags_block=tags_block,
        name=catalog_block.get("name", recipe.id),
        recipe_id=recipe.id,
        mirror_line=mirror_line,
        facts_block=facts_block,
        min_os=min_os_str,
        gate_a_status=report["gate_a"]["status"],
        evaluation_block=evaluation_block,
        attribution=attribution,
        collection_link=collection_link,
        recipe_url=f"{FABRIC_REPO_URL}/blob/main/recipes/{recipe.id}.yaml",
        tool=manifest.get("tool", conv["tool"]),
        tool_version=manifest.get("tool_version") or "(version not reported)",
        precision=conv["precision"],
        quantization=conv["quantization"],
        date=today(),
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
    if recipe.data["upstream"]["license_terms"] == "restricted":
        # Hard stop, NOT ackable: a restricted upstream's converted weights are
        # never redistributable. Index the upstream + ship the recipe instead
        # (the reproducible recipe is fabric's own code and ships freely).
        err(
            "upstream license is restricted — fabric refuses to republish converted "
            "weights of a restricted upstream (no --acknowledge bypass). Index the "
            "upstream and ship the reproducible recipe instead."
        )
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
        license_files = fetch_upstream_license(
            upstream["hf_repo"], revision_pin, staging,
            root=root, declared_license=upstream.get("license"))
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
    if getattr(args, "and_register", False):
        return _chain_register(recipe.id, getattr(args, "catalog_path", None))
    print(f"next: coreai-fabric register {recipe.id} --catalog-path ../coreai-catalog --dry-run")
    return 0


def cmd_mirror(args) -> int:
    """S3 — mirror a published, canonical repo into a distribution org
    (default `coreai-community`), preserving your namespace as the source of
    truth. This is the "mirror depois" step: your namespace owns the canonical
    repo; the org copy is a discovery surface. Records publish.mirror_namespace
    so the next register emits the machine-readable mirrors[] link."""
    root = find_root()
    recipe = find_recipe(args.id, root)
    published = recipe.data.get("published")
    if not published or not published.get("hf_repo"):
        err(f"{recipe.id} has no published block — publish to your namespace first, then mirror")
        return 1
    source = published["hf_repo"]
    repo_name = source.split("/", 1)[1]
    target = f"{args.to}/{repo_name}"
    if source == target:
        err(f"source and target are the same repo ({source}) — nothing to mirror")
        return 1
    print(f"mirror: {source}  ->  {target}")
    if args.dry_run:
        print("  dry-run: would create the target repo (private→public), copy the bundle +")
        print("  card + LICENSE + reports from the canonical repo, and record")
        print(f"  publish.mirror_namespace: {args.to} so register emits the mirrors[] link.")
        return 0
    import tempfile

    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    api.create_repo(target, repo_type="model", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        local = snapshot_download(source, local_dir=tmp)
        api.upload_folder(repo_id=target, folder_path=local,
                          commit_message=f"Mirror of {source}")
    _set_private(api, target, False)
    ok(f"mirrored -> https://huggingface.co/{target}")
    recipe.data.setdefault("publish", {})["mirror_namespace"] = args.to
    write_yaml(recipe.path, recipe.data)
    ok(f"recorded publish.mirror_namespace: {args.to} ({recipe.path.name})")
    print(f"next: coreai-fabric register {recipe.id} --catalog-path <path>  # emits mirrors[]")
    return 0


def _catalog_clone() -> Path | None:
    """Clone or fast-forward the catalog into ~/.cache/coreai-fabric/catalog so
    `publish --and-register` works without an explicit --catalog-path."""
    import subprocess

    from . import CATALOG_REPO_URL
    dest = Path.home() / ".cache" / "coreai-fabric" / "catalog"
    if (dest / ".git").is_dir():
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"],
                       capture_output=True, text=True)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["git", "clone", "--depth", "1", f"{CATALOG_REPO_URL}.git", str(dest)],
                       capture_output=True, text=True)
    return dest if r.returncode == 0 else None


def _chain_register(recipe_id: str, catalog_path) -> int:
    """S1 — the seamless publish→index handoff: after a successful publish, run
    the register flow so the catalog PR opens in the same command. Opt-in via
    --and-register; the register step still needs an authenticated `gh`."""
    from types import SimpleNamespace

    from .register import cmd_register
    if not catalog_path:
        catalog_path = _catalog_clone()
        if catalog_path is None:
            warn(f"published, but --and-register could not obtain a catalog clone; "
                 f"run `coreai-fabric register {recipe_id} --catalog-path <path>` yourself")
            return 0
        print(f"--and-register: using catalog clone at {catalog_path}")
    print(f"--and-register: registering {recipe_id} into the catalog ...")
    return cmd_register(SimpleNamespace(
        id=recipe_id, catalog_path=str(catalog_path), dry_run=False, mark_merged=False))


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
