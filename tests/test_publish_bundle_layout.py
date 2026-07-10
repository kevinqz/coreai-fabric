"""Publishing the Apple `coreai.llm.export` SPLIT bundle layout.

Apple's exporter writes the LanguageBundle descriptor (`metadata.json` with
`assets.main` + `language`) and the `tokenizer/` dir as SIBLINGS of the
`<id>.aimodel/` asset — not inside it. Publish must ship all three (so the HF
bundle loads via `LanguageBundle(from:)`), while excluding build cruft
(conversion-manifest.json, parity-report.json, bench-out/, publish-staging/).
"""
from __future__ import annotations

import json
from pathlib import Path

from coreai_fabric.publish import assert_bundle_content, split_bundle_members
from coreai_fabric.recipes import Recipe


def _make_split_bundle(tmp_path: Path) -> Path:
    """build/m/ with descriptor + m.aimodel/ + tokenizer/ + build cruft."""
    outdir = tmp_path / "build" / "m"
    outdir.mkdir(parents=True)
    (outdir / "metadata.json").write_text(json.dumps({
        "metadata_version": "0.2", "kind": "llm", "name": "m",
        "assets": {"main": "m.aimodel"},
        "language": {"tokenizer": "Org/M", "vocab_size": 100, "embedded_tokenizer": True},
    }))
    asset = outdir / "m.aimodel"
    asset.mkdir()
    (asset / "main.mlirb").write_bytes(b"\x00" * 16)
    (asset / "main.hash").write_text("deadbeef")
    (asset / "metadata.json").write_text(json.dumps(
        {"assetVersion": "2.0", "producer": "coreai-core"}))  # asset-only inner meta
    tok = outdir / "tokenizer"
    tok.mkdir()
    (tok / "tokenizer.json").write_text("{}")
    (tok / "merges.txt").write_text("")
    # Build cruft that must NOT be published:
    (outdir / "conversion-manifest.json").write_text("{}")
    (outdir / "parity-report.json").write_text("{}")
    (outdir / "bench-out").mkdir()
    (outdir / "bench-out" / "trials.jsonl").write_text("{}\n")
    return asset  # bundle_path convention: build/m/m.aimodel


def _make_single_dir_bundle(tmp_path: Path) -> Path:
    """Fabric-driver layout: the descriptor lives INSIDE m.aimodel/."""
    asset = tmp_path / "build" / "m" / "m.aimodel"
    asset.mkdir(parents=True)
    (asset / "metadata.json").write_text(json.dumps({
        "metadata_version": "0.2", "kind": "llm", "name": "m",
        "assets": {"main": "m.aimodel"}, "language": {"tokenizer": "Org/M"},
    }))
    (asset / "main.mlirb").write_bytes(b"\x00" * 16)
    (asset / "main.hash").write_text("deadbeef")
    return asset


def test_split_members_are_descriptor_asset_tokenizer(tmp_path):
    bundle = _make_split_bundle(tmp_path)
    members = split_bundle_members(bundle)
    assert members is not None
    names = {m.name for m in members}
    assert names == {"metadata.json", "m.aimodel", "tokenizer"}


def test_split_members_exclude_build_cruft(tmp_path):
    bundle = _make_split_bundle(tmp_path)
    names = {m.name for m in split_bundle_members(bundle)}
    for cruft in ("conversion-manifest.json", "parity-report.json", "bench-out"):
        assert cruft not in names


def test_single_dir_layout_is_not_split(tmp_path):
    bundle = _make_single_dir_bundle(tmp_path)
    assert split_bundle_members(bundle) is None


def _recipe(bundle_files):
    return Recipe(Path("recipes/m.yaml"),
                  {"id": "m", "expected": {"bundle_files": bundle_files}})


def test_content_gate_allows_tokenizer_and_descriptor(tmp_path):
    # A staged split bundle: descriptor + asset dir + tokenizer/ — all allowed.
    stage = tmp_path / "stage"
    (stage / "m.aimodel").mkdir(parents=True)
    (stage / "metadata.json").write_text("{}")
    (stage / "m.aimodel" / "main.mlirb").write_bytes(b"\x00")
    (stage / "m.aimodel" / "main.hash").write_text("x")
    (stage / "m.aimodel" / "metadata.json").write_text("{}")
    (stage / "tokenizer").mkdir()
    (stage / "tokenizer" / "tokenizer.json").write_text("{}")
    (stage / "tokenizer" / "vocab.json").write_text("{}")
    assert assert_bundle_content(stage, _recipe(["metadata.json", "main.mlirb", "main.hash"])) == []


def test_content_gate_still_refuses_stray_weights(tmp_path):
    stage = tmp_path / "stage"
    (stage / "m.aimodel").mkdir(parents=True)
    (stage / "metadata.json").write_text("{}")
    (stage / "model.safetensors").write_bytes(b"\x00")  # stray derivative weights
    offending = assert_bundle_content(stage, _recipe(["metadata.json"]))
    assert "model.safetensors" in offending
