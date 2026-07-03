"""`coreai-fabric convert` — execute the conversion via the Apple toolchain adapter.

Honesty note (read before trusting this module): the exact CLI interface of
Apple's converter (apple/coreai-torch) has NOT been verified offline. The
adapter below isolates that single assumption in `build_command()` — one
function, explicitly TODO-marked. Everything around it (toolchain discovery,
manifest, honest failure) is real and tested. Conversion requires macOS with
the Apple Core AI toolchain installed; it never runs in CI.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from . import hf
from .recipes import Recipe, find_recipe
from .util import err, find_root, ok, utc_now_iso, warn, write_yaml

TOOLCHAIN_INSTALL_HINT = """\
The Apple Core AI converter was not found on PATH.

To convert models you need macOS on Apple Silicon with Apple's toolchain:
  1. Apple's converter and export recipes live in the apple/coreai-torch and
     apple/coreai-models GitHub repos — follow their install instructions.
  2. Ensure the converter executable (default: coreai-torch) is on PATH,
     or set COREAI_FABRIC_TOOL=/path/to/converter.
  3. Re-run: coreai-fabric convert <id>

Fabric cannot install the Apple toolchain for you, and CI never converts.
"""


def bundle_path(root: Path, recipe: Recipe) -> Path:
    return root / "build" / recipe.id / f"{recipe.id}.aimodel"


def manifest_path(root: Path, recipe: Recipe) -> Path:
    return root / "build" / recipe.id / "conversion-manifest.json"


def build_command(recipe: Recipe, tool: str, output: Path) -> list[str]:
    """Construct the converter invocation from recipe params.

    TODO(toolchain): this argument layout is fabric's recipe contract mapped
    onto an assumed `<tool> export` interface. It has not been verified
    against a real apple/coreai-torch install (unavailable offline). When the
    toolchain is available, confirm flag names here and bump
    conversion.min_tool_version in the recipes accordingly. Use
    --print-command to inspect without executing.
    """
    conv = recipe.data["conversion"]
    upstream = recipe.data["upstream"]
    cmd = [
        tool,
        "export",
        upstream["hf_repo"],
        "--output",
        str(output),
        "--precision",
        str(conv["precision"]),
        "--quantization",
        str(conv["quantization"]),
        "--compute-units",
        str(conv["compute_units"]),
    ]
    if upstream.get("revision"):
        cmd += ["--revision", upstream["revision"]]
    for key, value in sorted((conv.get("args") or {}).items()):
        cmd += [f"--{key}", str(value)]
    return cmd


def tool_version(tool: str) -> str | None:
    """Best-effort `<tool> --version` capture. Never fabricated: returns None
    when the tool does not report one."""
    try:
        proc = subprocess.run(
            [tool, "--version"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = (proc.stdout or proc.stderr).strip()
    return out.splitlines()[0] if out else None


def cmd_convert(args) -> int:
    root = find_root()
    recipe = find_recipe(args.id, root)

    tool = os.environ.get("COREAI_FABRIC_TOOL") or recipe.data["conversion"]["tool"]
    output = bundle_path(root, recipe)
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
        "command": cmd,
        "input": {
            "hf_repo": upstream["hf_repo"],
            "revision": resolved_revision,
        },
        "started_at": started,
        "finished_at": finished,
        "output_bundle": str(output.relative_to(root)),
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
