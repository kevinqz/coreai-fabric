"""``coreai-fabric run <id>`` — RFC Phase 1 (F8): the single failure-capture choke point.

Today ``convert`` PRINTS the venv-aware converter invocation and refuses to drive
script tools by design; ``run`` executes that exact invocation and instruments
it. It is the choke point the weakness-mining loop needs — THE DRIVERS DO NOT
CHANGE.

Captured per attempt -> appended to ``attempts/<id>.jsonl`` (COMMITTED, including
failures). Failed exports previously left zero structured trace; fabric never saw
the 36/53 script-tool runs. Now every attempt — exit 0 or not — is durable,
clusterable by ``error_signature``, and linkable to its eventual Gate-B outcome.

The record:
    {ts, recipe, stage, tool, exit, toolchain{...}, error_signature, error_tail,
     envelope{precision, quantization, ...}, outcome}

``outcome`` in {converted, blocked, failed, parity_below_threshold} — the
``error_signature`` table decides whether a non-zero exit is an external ceiling
(blocked) or a fixable attempt (failed).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .convert import (
    build_command,
    bundle_path,
    converter_stack_versions,
    is_script_tool,
    script_tool_hint,
)
from .error_signatures import outcome
from .recipes import Recipe, find_recipe
from .util import err, find_root, ok, utc_now_iso, warn

#: How much of the stderr tail to keep in the record (the diagnostic, not the flood).
ERROR_TAIL_BYTES = 2000

#: Records that name a recipe's converter invocation that fabric does not drive
#: (per-model PEP 723 scripts). `run` still captures their exit + stderr when the
#: user runs the printed command through the wrapper.


def attempts_path(root: Path, recipe: Recipe) -> Path:
    return root / "attempts" / f"{recipe.id}.jsonl"


def _envelope(recipe: Recipe) -> dict:
    """The conversion envelope that identifies WHAT was attempted (F12: a fact is
    a point in (graph, shape, precision, host-contract) space)."""
    conv = recipe.data.get("conversion", {})
    env = {
        "precision": conv.get("precision"),
        "quantization": conv.get("quantization"),
        "tool": conv.get("tool"),
    }
    action = conv.get("action") or {}
    graphs = action.get("graphs")
    if isinstance(graphs, list) and graphs:
        env["graphs"] = [str(g.get("name", "")) for g in graphs if isinstance(g, dict)]
    sampling = (action.get("sampling") or {}).get("num_steps")
    if isinstance(sampling, int):
        env["num_steps"] = sampling
    return {k: v for k, v in env.items() if v is not None}


def _build_attempt_record(
    recipe: Recipe,
    *,
    stage: str,
    tool: str,
    cmd: list[str] | str,
    proc: subprocess.CompletedProcess | None,
    exit_code: int,
    error_text: str,
) -> dict:
    out, sig = outcome(exit_code, error_text)
    return {
        "ts": utc_now_iso(),
        "recipe": recipe.id,
        "stage": stage,
        "tool": tool,
        "command": cmd if isinstance(cmd, str) else " ".join(cmd),
        "exit": exit_code,
        "toolchain": converter_stack_versions(),
        "error_signature": sig,
        "error_tail": error_text[-ERROR_TAIL_BYTES:].strip() if error_text else "",
        "envelope": _envelope(recipe),
        "outcome": out,
    }


def append_attempt(root: Path, recipe: Recipe, record: dict) -> Path:
    """Append one attempt record to attempts/<id>.jsonl (created if absent)."""
    path = attempts_path(root, recipe)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return path


def _run_invocation(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run the converter invocation, capturing stdout+stderr combined for the
    error-signature classifier."""
    return subprocess.run(cmd, capture_output=True, text=True)


def cmd_run(args) -> int:
    root = find_root()
    recipe = find_recipe(args.id, root)

    tool = os.environ.get("COREAI_FABRIC_TOOL") or recipe.data["conversion"]["tool"]
    output = bundle_path(root, recipe)

    if is_script_tool(tool):
        # Per-model PEP 723 script: fabric documents the verified `uv run` line
        # but does not drive it. `run` captures the attempt when the user routes
        # the printed command through it via -- (the drivers do not change).
        hint = script_tool_hint(recipe, tool)
        if args.print_command:
            print(hint)
            return 0
        err(f"converter '{tool}' is a per-model script — fabric does not drive it.")
        print(hint)
        print("\nTo capture this run in attempts/, route the printed `uv run` line through:")
        print(f"  coreai-fabric run {recipe.id} -- <uv-run-command...>")
        return 1

    cmd = build_command(recipe, tool, output)
    if args.print_command:
        print(" ".join(cmd))
        return 0

    tool_path = shutil.which(tool)
    if not tool_path:
        record = _build_attempt_record(
            recipe, stage="convert", tool=tool, cmd=cmd, proc=None,
            exit_code=127, error_text=f"converter '{tool}' not found on PATH",
        )
        path = append_attempt(root, recipe, record)
        err(f"converter '{tool}' not found on PATH")
        warn(f"attempt recorded (outcome={record['outcome']}): {path.relative_to(root)}")
        return 1

    print(f"running: {' '.join(cmd)}")
    proc = _run_invocation(cmd)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    record = _build_attempt_record(
        recipe, stage="convert", tool=tool, cmd=cmd, proc=proc,
        exit_code=proc.returncode, error_text=combined if proc.returncode else "",
    )
    path = append_attempt(root, recipe, record)
    print(f"attempt recorded: {path.relative_to(root)} (exit={proc.returncode}, "
          f"outcome={record['outcome']}, signature={record['error_signature']})")

    if proc.returncode != 0:
        err(f"converter exited {proc.returncode}; see {path.relative_to(root)}")
        # Surface the distilled error class prominently.
        if record["error_signature"] != "unclassified":
            warn(f"failure class: {record['error_signature']} (outcome={record['outcome']})")
        return proc.returncode

    ok(f"conversion succeeded for {recipe.id}")
    print(f"next: coreai-fabric verify {recipe.id}")
    return 0
