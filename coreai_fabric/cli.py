#!/usr/bin/env python3
"""coreai-fabric CLI — the conversion fabric for Apple Core AI.

Usage:
  coreai-fabric new Qwen/Qwen3-0.6B
  coreai-fabric validate
  coreai-fabric validate qwen3-0.6b
  coreai-fabric convert qwen3-0.6b
  coreai-fabric verify qwen3-0.6b
  coreai-fabric publish qwen3-0.6b
  coreai-fabric register qwen3-0.6b --catalog-path ../coreai-catalog --dry-run
  coreai-fabric list
  coreai-fabric status qwen3-0.6b
"""
from __future__ import annotations

import argparse
import sys

from . import __version__
from .recipes import find_recipe, load_all_recipes, recipe_schema, validate_recipe
from .util import BOLD, DIM, GREEN, RED, RESET, YELLOW, err, find_root

NEXT_STEP = {
    "draft": "coreai-fabric convert {id}",
    "converted": "coreai-fabric verify {id}",
    "verified": "coreai-fabric publish {id}",
    "published": "coreai-fabric register {id} --catalog-path ../coreai-catalog",
    "registered": "(done — indexed by coreai-catalog)",
}


def cmd_validate(args) -> int:
    root = find_root()
    schema = recipe_schema(root)
    recipes = [find_recipe(args.id, root)] if args.id else load_all_recipes(root)
    if not recipes:
        err("no recipes found under recipes/")
        return 1

    all_issues = []
    for recipe in recipes:
        all_issues.extend(validate_recipe(recipe, schema))

    errors = [i for i in all_issues if i.severity == "error"]
    warnings = [i for i in all_issues if i.severity == "warning"]
    for issue in errors + warnings:
        color = RED if issue.severity == "error" else YELLOW
        print(f"{color}{issue.render()}{RESET}")
    print(
        f"validated {len(recipes)} recipe(s): "
        f"{RED if errors else GREEN}{len(errors)} error(s){RESET}, "
        f"{YELLOW if warnings else ''}{len(warnings)} warning(s){RESET if warnings else ''}"
    )
    return 1 if errors else 0


def cmd_list(args) -> int:
    root = find_root()
    recipes = load_all_recipes(root)
    if not recipes:
        print("no recipes yet — start with: coreai-fabric new Qwen/Qwen3-0.6B")
        return 0
    rows = [("ID", "STATUS", "UPSTREAM", "LICENSE", "PUBLISHED")]
    for r in recipes:
        upstream = r.data.get("upstream", {})
        published = r.data.get("published", {})
        rows.append(
            (
                r.id,
                r.status,
                upstream.get("hf_repo", "?"),
                f"{upstream.get('license', '?')} ({upstream.get('license_terms', '?')})",
                published.get("hf_repo", "-"),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for n, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        print(f"{BOLD}{line}{RESET}" if n == 0 else line)
    return 0


def cmd_status(args) -> int:
    root = find_root()
    if args.id:
        recipes = [find_recipe(args.id, root)]
    else:
        recipes = load_all_recipes(root)
    stages = ["draft", "converted", "verified", "published", "registered"]
    for r in recipes:
        marker_line = " -> ".join(
            f"{GREEN}[{s}]{RESET}" if stages.index(r.status) >= i else f"{DIM}{s}{RESET}"
            for i, s in enumerate(stages)
        )
        print(f"{BOLD}{r.id}{RESET}  {marker_line}")
        print(f"  upstream: {r.data.get('upstream', {}).get('hf_repo', '?')}")
        if r.data.get("published"):
            p = r.data["published"]
            print(f"  published: https://huggingface.co/{p['hf_repo']} @ {p['revision'][:12]}")
        print(f"  next: {NEXT_STEP[r.status].format(id=r.id)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coreai-fabric",
        description="Recipes in, provenance-verified .aimodel out, indexed by coreai-catalog.",
    )
    parser.add_argument("--version", action="version", version=f"coreai-fabric {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="scaffold a draft recipe from an upstream HF repo")
    p_new.add_argument("upstream_hf_repo", help="upstream Hugging Face repo id (owner/name)")
    p_new.add_argument("--id", help="recipe id (default: derived from the repo name)")
    p_new.add_argument("--namespace", default=None,
                       help="the publisher's OWN HF namespace to publish into "
                       "(default: your logged-in HF user via `hf whoami`). Fabric "
                       "refuses to scaffold into a known shared org (e.g. "
                       "coreai-community) unless you pass --i-am-mirroring — your "
                       "own namespace is the source of truth; a shared org is a mirror.")
    p_new.add_argument("--i-am-mirroring", action="store_true",
                       help="acknowledge you are targeting a SHARED org (mirror), "
                       "not your own namespace — required to scaffold into one.")
    p_new.add_argument("--repo-name",
                       help="target HF repo name (default: <UpstreamModelName>-CoreAI)")
    p_new.add_argument("--collection", default="CoreAI · Apple on-device",
                       help="HF Collection title (under the namespace) to add the "
                       "published model to, grouping your CoreAI work "
                       "(default: 'CoreAI · Apple on-device'; pass '' to disable)")
    p_new.add_argument("--tool", default="coreai-fabric-llm-export",
                       help="converter executable (default: coreai-fabric-llm-export, "
                       "fabric's verified driver over coreai-torch; Apple's "
                       "coreai.llm.export from an apple/coreai-models checkout "
                       "accepts the same flag layout)")
    p_new.add_argument("--apple-registry-name",
                       help="PRODUCTION path: the Apple model-registry short-name "
                       "(e.g. qwen3-0.6b, from `coreai.model.registry "
                       "--list-models`). Only valid with --tool coreai.llm.export; "
                       "makes convert drive `coreai.llm.export <short-name>` so "
                       "Apple's TESTED compression preset resolves (the KV-cache "
                       "chat asset). precision/quantization then document the preset.")
    p_new.add_argument("--variant", choices=["int8"],
                       help="scaffold the VERIFIED int8 lane (the absmax/per-block-32 "
                       "compression config that measured ~lossless on Qwen3-0.6B) instead "
                       "of Apple's lossy 4bit preset — the SotA default for an LLM. Only "
                       "with --tool coreai.llm.export; drop --apple-registry-name. Lands "
                       "as the int8/ tier and defaults gate_b to greedy_parity.")
    p_new.add_argument("--precision", default="float16",
                       help="passed as --compute-precision (verified vocabulary: "
                       "float16, bfloat16, float32)")
    p_new.add_argument("--quantization", default="none",
                       help="passed as --compression (none, or an Apple preset "
                       "when using coreai.llm.export)")
    p_new.add_argument("--platform", default="macOS",
                       help="target platform arg for the converter (default: macOS)")
    p_new.add_argument("--bundle-file", action="append",
                       help="expected bundle file (repeatable; default: the verified "
                       ".aimodel inventory metadata.json, main.mlirb, main.hash)")
    p_new.add_argument("--license", help="upstream license id (required with --offline)")
    p_new.add_argument("--license-terms", choices=["permissive", "weak_copyleft", "restricted", "review_required", "unknown"])
    p_new.add_argument("--pipeline-tag", help="override the upstream pipeline_tag")
    p_new.add_argument("--name", help="catalog display name")
    p_new.add_argument("--family", help="catalog model family")
    p_new.add_argument("--capability", action="append", help="catalog capability (repeatable)")
    p_new.add_argument("--input", action="append", help="input modality (repeatable)")
    p_new.add_argument("--output", action="append", help="output modality (repeatable)")
    p_new.add_argument("--notes", help="recipe notes")
    p_new.add_argument("--offline", action="store_true",
                       help="skip the HF API (requires --license)")
    p_new.add_argument("--force", action="store_true", help="overwrite an existing recipe")

    p_val = sub.add_parser("validate", help="validate recipe(s): schema + license triage, aggregated errors")
    p_val.add_argument("id", nargs="?", help="recipe id (default: all recipes)")

    p_conv = sub.add_parser("convert", help="convert upstream -> .aimodel via the Apple toolchain adapter")
    p_conv.add_argument("id")
    p_conv.add_argument("--print-command", action="store_true",
                        help="print the converter invocation without running it")

    p_ver = sub.add_parser("verify", help="Gate A (structure) + Gate B (numeric parity); writes parity-report.json")
    p_ver.add_argument("id")

    p_pub = sub.add_parser("publish", help="upload the bundle + model card to the publisher's own HF namespace")
    p_pub.add_argument("id")
    p_pub.add_argument("--dry-run", action="store_true", help="print the model card and target, upload nothing")
    p_pub.add_argument("--acknowledge-license-review", action="store_true",
                       help="confirm a human reviewed a review_required license")
    p_pub.add_argument("--allow-unverified-parity", action="store_true",
                       help="publish although Gate B is NOT_RUN (unmeasured; recorded honestly)")
    p_pub.add_argument("--publish-known-lossy-size-tier", action="store_true",
                       help="publish although Gate B FAILED (measured-lossy, e.g. int4 on Qwen) "
                       "as an explicit size-optimized tier — the card carries the measured number")
    p_pub.add_argument("--allow-missing-license-file", action="store_true",
                       help="publish even if no upstream LICENSE/NOTICE could be mirrored "
                       "(only for upstreams that genuinely ship none — fabric refuses by default)")
    p_pub.add_argument("--and-register", action="store_true",
                       help="S1 seamless flow: after a successful publish, register the model "
                       "into the catalog (opens the catalog PR — needs an authenticated `gh`). "
                       "Clones the catalog into ~/.cache/coreai-fabric/catalog if no path given.")
    p_pub.add_argument("--catalog-path", help="path to a coreai-catalog clone for --and-register "
                       "(default: auto-clone into ~/.cache/coreai-fabric/catalog)")

    p_mirror = sub.add_parser("mirror", help="mirror a published canonical repo into a distribution org (e.g. coreai-community)")
    p_mirror.add_argument("id")
    p_mirror.add_argument("--to", default="coreai-community",
                          help="target org for the mirror (default: coreai-community). Your "
                          "namespace stays the canonical source of truth.")
    p_mirror.add_argument("--dry-run", action="store_true",
                          help="print the source→target plan, copy nothing")

    p_reg = sub.add_parser("register", help="generate catalog entries and open a PR to kevinqz/coreai-catalog")
    p_reg.add_argument("id")
    p_reg.add_argument("--catalog-path", help="path to a coreai-catalog clone (schemas + PR staging)")
    p_reg.add_argument("--dry-run", action="store_true", help="print the generated YAML, change nothing")
    p_reg.add_argument("--mark-merged", action="store_true",
                       help="flip status published->registered after the catalog PR (catalog_pr) merges")

    sub.add_parser("list", help="recipe inventory with pipeline stage")

    p_stat = sub.add_parser("status", help="pipeline stage + next step per recipe")
    p_stat.add_argument("id", nargs="?", help="recipe id (default: all recipes)")

    # Tier 3: AOT compile (.aimodel → .aimodelc) — spec §3.4 future command.
    # Ahead-of-time compilation for iOS deployment of large models (>4B JIT causes jetsam).
    p_compile = sub.add_parser("compile",
        help="AOT-compile a .aimodel bundle to .aimodelc (Tier 3 — requires Core AI SDK compile API)")
    p_compile.add_argument("id", help="recipe id (the bundle to compile)")
    p_compile.add_argument("--target", choices=["ios", "macos"], default="ios",
        help="target platform for specialization (default: ios)")
    p_compile.add_argument("--dry-run", action="store_true",
        help="print the compile command without executing")

    # lerobot export — spec §17.4 wrapper command.
    p_lr = sub.add_parser("lerobot",
        help="LeRobot-specific commands (lerobot-coreai integration)")
    p_lr_sub = p_lr.add_subparsers(dest="lerobot_command")
    p_lr_export = p_lr_sub.add_parser("export",
        help="export a LeRobot policy: new → convert → verify → publish + lerobot-coreai.json")
    p_lr_export.add_argument("--upstream.repo", dest="upstream_repo", required=True,
        help="upstream LeRobot policy HF repo (e.g. lerobot/act_so100)")
    p_lr_export.add_argument("--policy.type", dest="policy_type", required=True,
        help="policy type (e.g. act, pi0, diffusion)")
    p_lr_export.add_argument("--robot.type", dest="robot_type", required=True,
        help="robot type (e.g. so100, so101)")
    p_lr_export.add_argument("--output.repo_id", dest="output_repo_id", required=True,
        help="output HF repo id (e.g. kevinqz/ACT-SO100-CoreAI)")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "new":
        from .scaffold import cmd_new

        return cmd_new(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "convert":
        from .convert import cmd_convert

        return cmd_convert(args)
    if args.command == "verify":
        from .verify import cmd_verify

        return cmd_verify(args)
    if args.command == "publish":
        from .publish import cmd_publish

        return cmd_publish(args)
    if args.command == "mirror":
        from .publish import cmd_mirror

        return cmd_mirror(args)
    if args.command == "register":
        from .register import cmd_register

        return cmd_register(args)
    if args.command == "list":
        return cmd_list(args)
    if args.command == "status":
        return cmd_status(args)
    if args.command == "compile":
        return cmd_compile(args)
    if args.command == "lerobot":
        return cmd_lerobot(args)
    raise AssertionError(f"unhandled command {args.command}")


def cmd_compile(args) -> int:
    """Tier 3: AOT-compile a .aimodel bundle to .aimodelc.

    Ahead-of-time compilation is required for iOS deployment of large models (>4B) where
    on-device JIT specialization overruns the per-process memory budget (jetsam kill).
    The zoo documents this as the "4B wall" (knowledge/aot-and-specialization.md).

    NOTE: This command requires the Core AI SDK's ahead-of-time compilation API, which
    is not yet available in the macOS 27.0 beta SDK. The command prints the intended
    invocation and exits with a clear message until the API ships.
    """
    print(f"compile: AOT-compile {args.id} for {args.target}")
    print()
    print("  STATUS: Not yet implemented — the Core AI SDK's ahead-of-time")
    print("  compilation API is not available in the macOS 27.0 beta SDK.")
    print()
    print("  When the API ships, this command will:")
    print(f"    1. Load the .aimodel bundle for recipe '{args.id}'")
    print(f"    2. Specialize for {args.target} (static shapes, per-device-class)")
    print(f"    3. Write the .aimodelc bundle alongside the .aimodel")
    print(f"    4. Update metadata.json to point at the compiled bundle")
    print()
    print("  For now, compile manually using the Apple exporter's AOT flag")
    print("  (see coreai-models knowledge/aot-and-specialization.md).")
    if args.dry_run:
        print("\n  (dry-run: no changes made)")
    return 1  # not yet implemented


def cmd_lerobot(args) -> int:
    """LeRobot-specific commands (lerobot-coreai integration, spec §17.4)."""
    if args.lerobot_command == "export":
        print(f"lerobot export: {args.upstream_repo} → {args.output_repo_id}")
        print()
        print("  This is a wrapper around the standard fabric pipeline:")
        print("    1. coreai-fabric new (scaffold recipe with lerobot: block)")
        print(f"       -- upstream: {args.upstream_repo}")
        print(f"       -- policy_type: {args.policy_type}")
        print(f"       -- robot_type: {args.robot_type}")
        print("    2. coreai-fabric convert")
        print("    3. coreai-fabric verify")
        print(f"    4. coreai-fabric publish (writes lerobot-coreai.json + card)")
        print("    5. coreai-fabric register")
        print()
        print("  Run each step manually for now. The wrapper will orchestrate")
        print("  them automatically in a future version.")
        return 0
    raise AssertionError(f"unhandled lerobot command {args.lerobot_command}")


if __name__ == "__main__":
    sys.exit(main())
