"""Doc-check: every `coreai-fabric ...` command shown in the agent-facing
docs must parse against the real argparse parser. Docs that teach commands
that do not exist are a P0 in an agent-first repo."""
from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from coreai_fabric.cli import build_parser

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = ["AGENTS.md", "README.md", "CONTRIBUTING.md"]

FENCE_RE = re.compile(r"```(?:bash|sh|console)?\n(.*?)```", re.DOTALL)


def _doc_commands() -> list[tuple[str, str]]:
    commands: list[tuple[str, str]] = []
    for doc in DOCS:
        text = (REPO_ROOT / doc).read_text()
        for block in FENCE_RE.findall(text):
            for line in block.splitlines():
                line = line.strip()
                if line.startswith("coreai-fabric "):
                    commands.append((doc, line))
    return commands


def test_docs_contain_commands():
    commands = _doc_commands()
    assert len(commands) >= 10, "agent docs lost their command examples"
    # The full pipeline must be demonstrated somewhere in the docs.
    joined = " ".join(c for _, c in commands)
    for verb in ("new", "validate", "convert", "verify", "publish", "register", "list", "status"):
        assert f"coreai-fabric {verb}" in joined, f"no doc example for '{verb}'"


@pytest.mark.parametrize("doc,command", _doc_commands(), ids=lambda v: str(v)[:60])
def test_every_documented_command_parses(doc, command):
    argv = shlex.split(command, comments=True)[1:]  # drop the program name
    parser = build_parser()
    try:
        parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code not in (0, None):
            pytest.fail(f"{doc}: documented command does not parse: {command}")
