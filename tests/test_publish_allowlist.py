"""RFC F10: the publish bundle content allowlist — refuses stray *.safetensors /
raw upstream slices before upload_folder ships them recursively."""
from __future__ import annotations

from pathlib import Path

from coreai_fabric.publish import assert_bundle_content


def _recipe(bundle_files=None):
    from coreai_fabric.recipes import Recipe
    return Recipe(
        path=Path("recipes/x.yaml"),
        data={"expected": {"bundle_files": bundle_files or ["metadata.json", "main.mlirb", "main.hash"]}},
    )


def _bundle(tmp_path: Path) -> Path:
    b = tmp_path / "build" / "x" / "x.aimodel"
    b.mkdir(parents=True)
    return b


def test_clean_bundle_passes(tmp_path):
    b = _bundle(tmp_path)
    for f in ("metadata.json", "main.mlirb", "main.hash"):
        (b / f).write_text("ok")
    assert assert_bundle_content(b, _recipe()) == []


def test_stray_safetensors_is_refused(tmp_path):
    b = _bundle(tmp_path)
    (b / "metadata.json").write_text("{}")
    (b / "raw_slice.safetensors").write_bytes(b"\x00" * 16)  # derivative NC data
    offending = assert_bundle_content(b, _recipe())
    assert "raw_slice.safetensors" in offending


def test_stray_pt_is_refused(tmp_path):
    b = _bundle(tmp_path)
    (b / "metadata.json").write_text("{}")
    (b / "leftover.pt").write_bytes(b"\x00")
    assert "leftover.pt" in assert_bundle_content(b, _recipe())


def test_declared_sidecar_is_allowed(tmp_path):
    # A recipe can declare extra bundle_files (e.g. a tokenizer); those pass.
    b = _bundle(tmp_path)
    (b / "metadata.json").write_text("{}")
    (b / "tokenizer.json").write_text("{}")
    recipe = _recipe(bundle_files=["metadata.json", "tokenizer.json"])
    assert assert_bundle_content(b, recipe) == []


def test_norm_stats_sidecar_is_allowed(tmp_path):
    # The action lane's norm_stats.json is a known sidecar.
    b = _bundle(tmp_path)
    (b / "metadata.json").write_text("{}")
    (b / "norm_stats.json").write_text("{}")
    assert assert_bundle_content(b, _recipe()) == []


def test_split_graph_programs_are_allowed(tmp_path):
    # The graph-split package layout (playbook T3): programs/*.aimodel parts.
    b = _bundle(tmp_path)
    (b / "metadata.json").write_text("{}")
    (b / "manifest.json").write_text("{}")
    (b / "programs").mkdir()
    (b / "programs" / "block0.aimodel").write_text("part")
    assert assert_bundle_content(b, _recipe()) == []
