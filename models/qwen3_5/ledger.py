"""Experiment ledger — the full-provenance record of every optimization attempt.

One immutable JSONL entry per attempt. Negative results count. Machine-readable,
committed, linked from the model card + catalog artifact provenance.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json

REQUIRED_FIELDS = (
    "id", "hypothesis", "target", "config_hash", "seed",
    "env_fingerprint", "deltas", "verdict", "why", "repro_cmd",
)
VERDICTS = ("kept", "rejected", "candidate")


@dataclass(frozen=True)
class LedgerEntry:
    id: str
    hypothesis: str
    target: dict
    config_hash: str
    seed: int
    env_fingerprint: dict
    deltas: dict
    verdict: str
    why: str
    repro_cmd: str

    @classmethod
    def from_dict(cls, d: dict) -> "LedgerEntry":
        for f in REQUIRED_FIELDS:
            if f not in d:
                raise ValueError(f"ledger entry missing required field: {f}")
        if d["verdict"] not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {d['verdict']!r}")
        return cls(**{f: d[f] for f in REQUIRED_FIELDS})

    def to_dict(self) -> dict:
        return asdict(self)


def record(entry: dict, *, path: str | Path) -> LedgerEntry:
    """Validate `entry` and append it as one JSONL line to `path`."""
    e = LedgerEntry.from_dict(entry)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(e.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return e
