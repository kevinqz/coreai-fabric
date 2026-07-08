"""Rebuild PR #34's branch (fabric/add-lingbot-vla-v2) off current catalog main so it's
clean/mergeable. Applies the single lingbot-vla-v2 entry + regenerates + runs the
catalog CI gates. No push (the caller force-pushes)."""
import sys, subprocess
from pathlib import Path
sys.path.insert(0, "/Users/kevinsaltarelli/Dev/Github/coreai-fabric")
from coreai_fabric.register import (
    find_root, find_recipe, build_model_entry, build_artifact_entry, build_source_record,
    _resolve_published_digests, _tool_version_from_manifest, _load_parity_report,
    _notes_suffix_from_report, _append_entry, _bump_artifact_count, _ensure_source_record,
)
CAT = Path.home() / "Dev/Github/coreai-catalog"
BRANCH = "fabric/add-lingbot-vla-v2"
root = find_root()


def git(*a, check=True):
    return subprocess.run(["git", "-C", str(CAT), *a], capture_output=True, text=True, check=check)


assert not git("status", "--porcelain").stdout.strip(), "catalog dirty"
git("checkout", "main")
git("checkout", "-B", BRANCH)   # reset branch to current main
r = find_recipe("lingbot-vla-v2", root)
files = _resolve_published_digests(r)
tv = _tool_version_from_manifest(root, r)
rep = _load_parity_report(root, r)
me = build_model_entry(r, files, notes_suffix=_notes_suffix_from_report(rep), report=rep)
ae = build_artifact_entry(r, files, tv)
_append_entry(CAT / "catalog.yaml", {"models": [me]})
_append_entry(CAT / "artifacts.yaml", {"artifacts": [ae]})
_bump_artifact_count(CAT / "artifacts.yaml")
_ensure_source_record(CAT / "sources.yaml", build_source_record())
print("applied lingbot-vla-v2", flush=True)
subprocess.run([sys.executable, "scripts/check_counts.py", "--fix"], cwd=CAT, capture_output=True, text=True)
gates = [(["scripts/validate.py"], "validate"), (["scripts/audit.py"], "audit"), (["scripts/generate.py"], "generate"),
         (["scripts/check_counts.py"], "check_counts"), (["scripts/doc_test.py"], "doc_test"),
         (["scripts/generate_templates.py", "--check"], "templates"), (["scripts/injection_lint.py"], "injection_lint")]
for argv, label in gates:
    if not (CAT / argv[0]).exists():
        print(f"  skip {label}", flush=True); continue
    p = subprocess.run([sys.executable, *argv], cwd=CAT, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"  GATE FAIL {label}:\n" + "\n".join((p.stdout + p.stderr).splitlines()[-8:]), flush=True); sys.exit(1)
    print(f"  gate {label}: ok", flush=True)
git("add", "-A")
git("commit", "-m", "feat: add lingbot-vla-v2 via coreai-fabric\n\nRebuilt off current main (was #34, cut from a stale base).")
print("committed on", BRANCH, "— ready to force-push", flush=True)
