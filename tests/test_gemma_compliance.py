"""Gemma redistribution compliance (Gemma Terms of Use §3.1/§3.2).

An adversarial 3-reviewer audit of the first Gemma-licensed publish found real gaps:
the mandated NOTICE string broken across a newline (so the verbatim substring was
absent), the §3.2 restrictions carried only as prose (no enforceable gating), a
holder-less card, and no Google non-endorsement clause. These tests lock the fixes so
the class of gap cannot regress. All hermetic — no network."""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from coreai_fabric.publish import fetch_upstream_license, render_model_card
from coreai_fabric.recipes import find_recipe

REPO_ROOT = Path(__file__).resolve().parents[1]
# §3.1(d): the NOTICE file must CONTAIN this exact string (single line).
MANDATED = ("Gemma is provided under and subject to the Gemma Terms of Use "
            "found at ai.google.dev/gemma/terms")
MANIFEST = {"tool": "models/pi0/export.py", "tool_version": "0.1", "input": {"revision": "abc123"}}
REPORT = {"gate_a": {"status": "passed"},
          "gate_b": {"metric": "action_parity", "status": "passed", "value": 0.996,
                     "environment": {"chip": "Apple Silicon"}, "num_steps": 10, "n_obs": 4}}


def _synthesize(monkeypatch) -> Path:
    """Force the no-upstream-license synthesis path without touching the network."""
    from huggingface_hub.utils import EntryNotFoundError

    def _boom(*a, **k):
        raise EntryNotFoundError("no license file (test)")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _boom)
    staging = Path(tempfile.mkdtemp())
    files = fetch_upstream_license("lerobot/pi0_base", "deadbeef", staging,
                                   root=REPO_ROOT, declared_license="gemma")
    assert set(files) == {"LICENSE", "NOTICE"}
    return staging


def test_notice_has_verbatim_mandated_string(monkeypatch):
    notice = (_synthesize(monkeypatch) / "NOTICE").read_text()
    # The exact single-line literal must appear as a contiguous substring — a wrapped
    # newline (the original bug) makes an automated/verbatim scan fail.
    assert MANDATED in notice, "mandated Gemma notice string is not a contiguous substring"
    assert "FLOW-DOWN" in notice, "§3.1 pass-through obligation missing from NOTICE"


def test_license_ships_full_gemma_terms(monkeypatch):
    lic = (_synthesize(monkeypatch) / "LICENSE").read_text()
    assert "Gemma Terms of Use" in lic
    assert "Section 3: DISTRIBUTION AND RESTRICTIONS" in lic
    assert "PaliGemma" in lic  # Appendix — the pi0 base is a PaliGemma derivative


def test_gemma_card_is_gated_and_surfaces_restrictions():
    card = render_model_card(REPO_ROOT, find_recipe("pi0-base-gemma", REPO_ROOT), MANIFEST, REPORT)
    fm = yaml.safe_load(card.split("---\n", 2)[1])
    # §3.1 enforceable provision: HF gating makes the recipient accept before download.
    assert "Gemma Terms of Use" in str(fm.get("extra_gated_fields")), "gemma repo must be gated"
    assert fm["license"] == "gemma"
    # §3.2 restrictions + attribution surfaced in the card BODY, not only in NOTICE.
    assert "Model Derivative of Gemma" in card
    assert "prohibited_use_policy" in card
    assert "Google LLC" in card                 # attribution holder (§ audit gap #4)
    assert "endorsed by Google" in card         # §4.2 non-endorsement (audit gap #5)


def test_non_gemma_card_is_not_gated():
    # apache/permissive recipes must NOT be gated — gating is a Gemma-only obligation.
    card = render_model_card(REPO_ROOT, find_recipe("folding-latest", REPO_ROOT), MANIFEST, REPORT)
    assert "extra_gated_fields" not in card
