"""RFC Phase 1b (F8): `coreai-fabric run` captures every conversion attempt —
including failures — into committed attempts/<id>.jsonl."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from coreai_fabric import run as run_mod
from coreai_fabric.recipes import Recipe


def _recipe(tmp_path: Path) -> Recipe:
    return Recipe(
        path=tmp_path / "recipes" / "x.yaml",
        data={
            "id": "x",
            "upstream": {"hf_repo": "o/m", "license": "mit", "license_terms": "permissive"},
            "conversion": {"tool": "coreai-fabric-llm-export", "quantization": "none",
                           "precision": "float16"},
            "expected": {"bundle_files": ["metadata.json"]},
            "parity": {"gate_a": {"checks": ["bundle_files_present"]},
                       "gate_b": {"metric": "graph_output_cosine", "threshold": 0.999,
                                  "tolerance": 0.0005}},
            "publish": {"hf_target_namespace": "o", "repo_name": "m-coreai"},
            "status": "draft",
        },
    )


def _args(recipe_id="x", print_command=False):
    return SimpleNamespace(id=recipe_id, print_command=print_command)


def test_run_appends_record_on_success(tmp_path, monkeypatch):
    recipe = _recipe(tmp_path)
    monkeypatch.setattr(run_mod, "find_root", lambda: tmp_path)
    monkeypatch.setattr(run_mod, "find_recipe", lambda _id, _root: recipe)
    monkeypatch.setattr(run_mod, "build_command", lambda r, t, o: ["echo", "hi"])
    monkeypatch.setattr(run_mod.shutil, "which", lambda _t: "/fake/echo")
    monkeypatch.setattr(run_mod, "_run_invocation",
                        lambda cmd: subprocess.CompletedProcess(cmd, 0, "done", ""))
    monkeypatch.setattr(run_mod, "converter_stack_versions", lambda: {})

    rc = run_mod.cmd_run(_args())
    assert rc == 0
    lines = (tmp_path / "attempts" / "x.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["outcome"] == "converted"
    assert rec["error_signature"] == "ok"
    assert rec["exit"] == 0
    assert rec["envelope"]["precision"] == "float16"


def test_run_appends_record_on_failure_with_signature(tmp_path, monkeypatch):
    # The whole point of F8: a FAILED run still leaves a structured trace.
    recipe = _recipe(tmp_path)
    monkeypatch.setattr(run_mod, "find_root", lambda: tmp_path)
    monkeypatch.setattr(run_mod, "find_recipe", lambda _id, _root: recipe)
    monkeypatch.setattr(run_mod, "build_command", lambda r, t, o: ["fake-export"])
    monkeypatch.setattr(run_mod.shutil, "which", lambda _t: "/fake/fake-export")
    monkeypatch.setattr(run_mod, "_run_invocation",
                        lambda cmd: subprocess.CompletedProcess(
                            cmd, 1, "",
                            "CompilerEngine: appleneuralengine Program load failure (0x10004)"))
    monkeypatch.setattr(run_mod, "converter_stack_versions", lambda: {})

    rc = run_mod.cmd_run(_args())
    assert rc == 1
    rec = json.loads((tmp_path / "attempts" / "x.jsonl").read_text().strip().splitlines()[0])
    assert rec["exit"] == 1
    assert rec["error_signature"] == "0x10004"
    assert rec["outcome"] == "blocked"  # ANE ceiling = external block
    assert "0x10004" in rec["error_tail"]


def test_run_records_missing_tool(tmp_path, monkeypatch):
    recipe = _recipe(tmp_path)
    monkeypatch.setattr(run_mod, "find_root", lambda: tmp_path)
    monkeypatch.setattr(run_mod, "find_recipe", lambda _id, _root: recipe)
    monkeypatch.setattr(run_mod, "build_command", lambda r, t, o: ["nope-export"])
    monkeypatch.setattr(run_mod.shutil, "which", lambda _t: None)

    rc = run_mod.cmd_run(_args())
    assert rc == 1
    rec = json.loads((tmp_path / "attempts" / "x.jsonl").read_text().strip().splitlines()[0])
    assert rec["exit"] == 127
    assert "not found" in rec["error_tail"]


def test_run_appends_multiple_records(tmp_path, monkeypatch):
    # The substrate is append-only: repeated attempts accumulate.
    recipe = _recipe(tmp_path)
    monkeypatch.setattr(run_mod, "find_root", lambda: tmp_path)
    monkeypatch.setattr(run_mod, "find_recipe", lambda _id, _root: recipe)
    monkeypatch.setattr(run_mod, "build_command", lambda r, t, o: ["echo"])
    monkeypatch.setattr(run_mod.shutil, "which", lambda _t: "/fake/echo")
    monkeypatch.setattr(run_mod, "_run_invocation",
                        lambda cmd: subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr(run_mod, "converter_stack_versions", lambda: {})

    run_mod.cmd_run(_args())
    run_mod.cmd_run(_args())
    lines = (tmp_path / "attempts" / "x.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_run_print_command_does_not_record(tmp_path, monkeypatch):
    recipe = _recipe(tmp_path)
    monkeypatch.setattr(run_mod, "find_root", lambda: tmp_path)
    monkeypatch.setattr(run_mod, "find_recipe", lambda _id, _root: recipe)
    monkeypatch.setattr(run_mod, "build_command", lambda r, t, o: ["echo", "cmd"])

    rc = run_mod.cmd_run(_args(print_command=True))
    assert rc == 0
    assert not (tmp_path / "attempts" / "x.jsonl").exists()
