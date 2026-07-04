"""`coreai-fabric convert` — execute the conversion via the Apple toolchain adapter.

Reality check (validated on real hardware — Apple M4 Max, macOS 26.6,
2026-07-03; see docs/toolchain-notes.md): Apple's `coreai-torch` PyPI package
is a Python LIBRARY (`TorchConverter`), NOT a CLI — the executable this module
was originally written against does not exist. The real converter executables
are:

- `coreai-fabric-llm-export` — fabric's own driver (ships with this repo,
  `pip install "coreai-fabric[convert]"`), built on the coreai-torch 0.4.1
  public API. Validated end-to-end on qwen3-0.6b.
- `coreai.llm.export` / `coreai.vlm.export` / `coreai.diffusion.export` —
  Apple's CLIs from the apple/coreai-models repo (NOT on PyPI; install from a
  checkout). Flag layout verified against the repo source at tag 0.1.0; both
  tools accept the layout emitted by `build_command`.
- Per-model PEP 723 scripts in apple/coreai-models (`models/<family>/export.py`,
  run via `uv run`) for non-LLM families (whisper, depth-anything, clip, ...).
  Their flags and output naming differ per script, so fabric refuses to drive
  them and prints the verified manual invocation instead.

macOS version guidance (verified): conversion AND Python-runtime execution
work on macOS 26.6 with the PyPI stack — macOS 27 is NOT required to convert.
Saved assets serialize with minimum_os v27 (the only OSVersion coreai-core
1.0.0b2 can emit), so DEPLOYING the artifact through the apple/coreai-models
Swift runners requires macOS/iOS 27+.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from . import hf
from .recipes import Recipe, find_recipe
from .util import err, find_root, ok, utc_now_iso, warn, write_yaml

#: Tools whose flag layout `build_command` emits. Verified 2026-07-03:
#: coreai-fabric-llm-export against a real run on this repo's driver;
#: coreai.llm.export against the apple/coreai-models source (tag 0.1.0).
LLM_EXPORT_TOOLS = {"coreai-fabric-llm-export", "coreai.llm.export"}

#: Only fabric's own driver supports upstream revision pinning. Apple's
#: coreai.llm.export has no --revision flag (verified against its argparse
#: source) — it converts whatever HF resolves at run time.
REVISION_CAPABLE_TOOLS = {"coreai-fabric-llm-export"}

TOOLCHAIN_INSTALL_HINT = """\
The converter executable was not found on PATH.

Verified reality of the Apple toolchain (macOS on Apple Silicon required):
  1. `coreai-torch` on PyPI is a LIBRARY, not a CLI — installing it alone
     gives you no executable.
  2. Fabric ships its own driver over that library:
         pip install "coreai-fabric[convert]"
     installs `coreai-fabric-llm-export` (validated on macOS 26.6 —
     macOS 27 is NOT needed to convert).
  3. Apple's PRODUCTION CLI (`coreai.llm.export` — KV-cache chat assets with
     Apple's tested compression presets) lives in the apple/coreai-models repo,
     which is NOT on PyPI. Install from a checkout:
         git clone https://github.com/apple/coreai-models
         pip install ./coreai-models/python
     Then use `tool: coreai.llm.export` + `apple_registry_name: <short-name>`
     in the recipe (list short-names with `coreai.model.registry --list-models
     --type llm`). Validated on macOS 26.6 — macOS 27 is only needed to RUN the
     asset, not to build it.
  4. Or set COREAI_FABRIC_TOOL=/path/to/converter.

Fabric cannot install the Apple toolchain for you, and CI never converts.
"""

SCRIPT_TOOL_HINT = """\
This recipe's verified converter is a per-model PEP 723 script from an
apple/coreai-models checkout, not a PATH executable. Its flags and output
naming are script-specific, so fabric does not drive it. Run it manually:

    git clone https://github.com/apple/coreai-models
    uv run coreai-models/{tool} {script_args}--output-dir <dir>

then place the produced .aimodel at build/{id}/{id}.aimodel and continue with
`coreai-fabric verify {id}`. (The script pins its own converter stack —
coreai-torch — via its PEP 723 header.)
"""


def bundle_path(root: Path, recipe: Recipe) -> Path:
    return root / "build" / recipe.id / f"{recipe.id}.aimodel"


def manifest_path(root: Path, recipe: Recipe) -> Path:
    return root / "build" / recipe.id / "conversion-manifest.json"


def build_command(recipe: Recipe, tool: str, output: Path) -> list[str]:
    """Construct the converter invocation from recipe params.

    Verified layout (2026-07-03, real hardware + apple/coreai-models source):
    `<tool> <hf_repo> --output-dir <build> --output-name <id>
    --compute-precision <precision> --compression <quantization> --overwrite
    [--revision <sha>] [--<arg> <value> ...]`

    Both `coreai-fabric-llm-export` and Apple's `coreai.llm.export` place the
    bundle at `<output-dir>/<output-name>/<output-name>.aimodel`, which is
    exactly fabric's `build/<id>/<id>.aimodel` when output-dir=build and
    output-name=<id>.
    """
    conv = recipe.data["conversion"]
    upstream = recipe.data["upstream"]
    out_dir = output.parent.parent  # build/<id>/<id>.aimodel -> build/
    registry_name = conv.get("apple_registry_name")
    compression_config = conv.get("compression_config")

    # HIGH-FIDELITY (int8) path: a custom coreai-opt quantization_config YAML.
    # Root-cause study (2026-07-03): Apple's macOS registry preset is int4 with
    # symmetric_with_clipping, which is genuinely lossy on Qwen (79% argmax);
    # int8 absmax/per-block-32 (no clipping) is ~lossless (96% argmax / 100%
    # top-5). This path passes --compression-config so a recipe can ship the
    # int8 (or any custom) tier. Needs --experimental + --compute-precision.
    if tool == "coreai.llm.export" and compression_config:
        cfg_path = out_dir.parent / compression_config  # repo root = build/'s parent
        cmd = [
            tool, upstream["hf_repo"],
            "--experimental",
            "--compute-precision", str(conv["precision"]),
            "--compression-config", str(cfg_path),
            "--output-dir", str(out_dir),
            "--output-name", recipe.id,
            "--overwrite",
        ]
        for key, value in sorted((conv.get("args") or {}).items()):
            cmd += [f"--{key}", str(value)]
        return cmd

    # PRODUCTION path: Apple's coreai.llm.export with a registry short-name
    # auto-resolves the TESTED compression preset for that model (e.g. qwen3-0.6b
    # -> 4bit/float16/8192), producing the KV-cache chat asset that passes parity.
    # Do NOT pass --compute-precision/--compression here — the preset must win.
    if tool == "coreai.llm.export" and registry_name:
        cmd = [
            tool,
            registry_name,
            "--output-dir", str(out_dir),
            "--output-name", recipe.id,
            "--overwrite",
        ]
        for key, value in sorted((conv.get("args") or {}).items()):
            cmd += [f"--{key}", str(value)]
        return cmd

    cmd = [
        tool,
        upstream["hf_repo"],
        "--output-dir",
        str(out_dir),
        "--output-name",
        recipe.id,
        "--compute-precision",
        str(conv["precision"]),
        "--compression",
        str(conv["quantization"]),
        "--overwrite",
    ]
    # coreai.llm.export refuses a raw HF ID (not a registry short-name) without
    # --experimental (verified against its argparse: "Allow exporting models
    # without a registry preset. Requires --compute-precision.").
    if tool == "coreai.llm.export":
        cmd.append("--experimental")
    if upstream.get("revision") and tool in REVISION_CAPABLE_TOOLS:
        cmd += ["--revision", upstream["revision"]]
    for key, value in sorted((conv.get("args") or {}).items()):
        cmd += [f"--{key}", str(value)]
    return cmd


def is_script_tool(tool: str) -> bool:
    """Per-model export scripts (models/<family>/export.py in an
    apple/coreai-models checkout) are documented in recipes but cannot be
    driven by fabric's uniform flag layout."""
    return tool.endswith(".py")


def script_tool_hint(recipe: Recipe, tool: str) -> str:
    args = recipe.data["conversion"].get("args") or {}
    rendered = "".join(f"--{k} {v} " for k, v in sorted(args.items()))
    return SCRIPT_TOOL_HINT.format(tool=tool, script_args=rendered, id=recipe.id)


def tool_version(tool: str) -> str | None:
    """Best-effort `<tool> --version` capture. Never fabricated: returns None
    when the tool does not report one (verified: Apple's coreai.llm.export has
    no --version flag and prints usage to stderr — that is not a version)."""
    try:
        proc = subprocess.run(
            [tool, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (proc.stdout or proc.stderr).strip()
    if not out or out.startswith("usage:") or proc.returncode != 0:
        return None
    return out.splitlines()[0]


def converter_stack_versions() -> dict:
    """Versions of the underlying conversion stack importable in THIS
    environment (informational; the tool may run in another env). Only
    reports what is actually installed — never fabricated."""
    from importlib import metadata

    versions = {}
    for dist in ("coreai-torch", "coreai-core", "torch", "transformers"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return versions


def cmd_convert(args) -> int:
    root = find_root()
    recipe = find_recipe(args.id, root)

    tool = os.environ.get("COREAI_FABRIC_TOOL") or recipe.data["conversion"]["tool"]
    output = bundle_path(root, recipe)

    if is_script_tool(tool):
        if args.print_command:
            print(script_tool_hint(recipe, tool))
            return 0
        err(f"converter '{tool}' is a per-model script, not a PATH executable")
        print(script_tool_hint(recipe, tool))
        return 1

    cmd = build_command(recipe, tool, output)

    if args.print_command:
        print(" ".join(cmd))
        return 0

    tool_path = shutil.which(tool)
    if not tool_path:
        err(f"converter '{tool}' not found on PATH")
        print(TOOLCHAIN_INSTALL_HINT)
        return 1

    upstream = recipe.data["upstream"]
    resolved_revision = upstream.get("revision")
    try:
        info = hf.model_info(upstream["hf_repo"], revision=resolved_revision)
        resolved_revision = info.get("sha") or resolved_revision
    except hf.HFError as exc:
        if not resolved_revision:
            err(f"cannot resolve upstream revision and none pinned in the recipe: {exc}")
            return 1
        warn(f"could not re-resolve upstream revision ({exc}); using pinned {resolved_revision}")

    if resolved_revision and tool not in REVISION_CAPABLE_TOOLS:
        warn(
            f"tool '{tool}' does not support upstream revision pinning "
            f"(verified); it will convert the repo's current head, not "
            f"{resolved_revision[:12]}. The manifest records what was requested."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now_iso()
    print(f"running: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    finished = utc_now_iso()
    if proc.returncode != 0:
        err(f"converter exited {proc.returncode}; build/{recipe.id}/ left as-is for inspection")
        return proc.returncode

    manifest = {
        "recipe_id": recipe.id,
        "tool": tool,
        "tool_path": tool_path,
        "tool_version": tool_version(tool),
        "converter_stack": converter_stack_versions(),
        "command": cmd,
        "input": {
            "hf_repo": upstream["hf_repo"],
            "revision": resolved_revision,
            "revision_pinned_by_tool": tool in REVISION_CAPABLE_TOOLS,
        },
        "started_at": started,
        "finished_at": finished,
        "output_bundle": str(output.relative_to(root)),
        # Verified: assets serialize with minimum_os v27 (coreai-core 1.0.0b2
        # exposes no other OSVersion). Conversion host: macOS 26.6 works.
        "asset_minimum_os": "27",
    }
    mpath = manifest_path(root, recipe)
    mpath.write_text(_dumps(manifest))
    ok(f"conversion manifest written: {mpath.relative_to(root)}")

    recipe.data["status"] = "converted"
    write_yaml(recipe.path, recipe.data)
    ok(f"recipe status -> converted ({recipe.path.name})")
    print(f"next: coreai-fabric verify {recipe.id}")
    return 0


def _dumps(data: dict) -> str:
    import json

    return json.dumps(data, indent=2) + "\n"
