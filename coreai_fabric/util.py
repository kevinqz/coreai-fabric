"""Shared helpers: root discovery, YAML IO, terminal formatting."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

_IS_TTY = sys.stdout.isatty()

BOLD = "\033[1m" if _IS_TTY else ""
DIM = "\033[2m" if _IS_TTY else ""
GREEN = "\033[32m" if _IS_TTY else ""
YELLOW = "\033[33m" if _IS_TTY else ""
RED = "\033[31m" if _IS_TTY else ""
RESET = "\033[0m" if _IS_TTY else ""


def find_root(start: Path | None = None) -> Path:
    """Locate the fabric repo root (the directory holding recipes/ + schema/).

    Resolution order: $COREAI_FABRIC_ROOT, then walk up from `start` (default
    cwd), then the installed package's parent (source checkout).
    """
    env = os.environ.get("COREAI_FABRIC_ROOT")
    if env:
        return Path(env).resolve()
    cur = (start or Path.cwd()).resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "recipes").is_dir() and (candidate / "schema" / "recipe.schema.json").is_file():
            return candidate
    pkg_parent = Path(__file__).resolve().parents[1]
    if (pkg_parent / "recipes").is_dir():
        return pkg_parent
    raise SystemExit(
        f"{RED}error:{RESET} not inside a coreai-fabric checkout "
        "(no recipes/ + schema/recipe.schema.json found walking up from cwd). "
        "Run from the repo root or set COREAI_FABRIC_ROOT."
    )


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def dump_yaml(data: dict) -> str:
    """Dump in the catalog's house style: key order preserved, indentless lists."""
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True, width=88)


def write_yaml(path: Path, data: dict, header: str | None = None) -> None:
    text = dump_yaml(data)
    if header:
        text = header.rstrip("\n") + "\n" + text
    path.write_text(text)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def ok(msg: str) -> None:
    print(f"{GREEN}ok:{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}warning:{RESET} {msg}")


def err(msg: str) -> None:
    print(f"{RED}error:{RESET} {msg}", file=sys.stderr)
