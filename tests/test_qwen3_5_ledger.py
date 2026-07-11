import json
from pathlib import Path
import pytest
from models.qwen3_5.ledger import LedgerEntry, record, REQUIRED_FIELDS


def _valid_entry() -> dict:
    return {
        "id": "exp-0001",
        "hypothesis": "int8lin body holds the eager gate",
        "target": {"model": "qwythos-9b", "weights_rev": "763f72f", "component": "gdn+full body"},
        "config_hash": "sha256:abc", "seed": 0,
        "env_fingerprint": {"py": "3.12", "transformers": "5.3.0", "coreai_torch": "0.4.1"},
        "deltas": {"gate": "24/24", "first_tok_cos": 0.9997},
        "verdict": "kept", "why": "matches Ornith int8lin",
        "repro_cmd": "python models/qwen3_5/parity.py --exp exp-0001",
    }


def test_ledgerentry_accepts_valid():
    e = LedgerEntry.from_dict(_valid_entry())
    assert e.id == "exp-0001"
    assert e.verdict == "kept"


def test_ledgerentry_rejects_missing_field():
    bad = _valid_entry()
    del bad["repro_cmd"]
    with pytest.raises(ValueError, match="missing required field: repro_cmd"):
        LedgerEntry.from_dict(bad)


def test_ledgerentry_rejects_bad_verdict():
    bad = _valid_entry()
    bad["verdict"] = "maybe"
    with pytest.raises(ValueError, match="verdict must be one of"):
        LedgerEntry.from_dict(bad)


def test_record_appends_jsonl(tmp_path):
    out = tmp_path / "provenance" / "exp.jsonl"
    record(_valid_entry(), path=out)
    second = _valid_entry(); second["id"] = "exp-0002"; second["verdict"] = "rejected"
    record(second, path=out)
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "exp-0001"
    assert json.loads(lines[1])["verdict"] == "rejected"


def test_record_rejects_invalid_before_write(tmp_path):
    out = tmp_path / "exp.jsonl"
    bad = _valid_entry(); del bad["why"]
    with pytest.raises(ValueError):
        record(bad, path=out)
    assert not out.exists()
