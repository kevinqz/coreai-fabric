#!/usr/bin/env bash
# Gate C driver — run the runtime loadability verifier on a converted bundle and
# emit a catalog-ready verdict line (for coreai-catalog/data/runtime-verifications.jsonl).
#
# Usage:
#   verify/gate-c.sh <model-id> <bundle-dir-or-.aimodel> <llm|graph> [--input "prompt"] [--append <jsonl>]
#
# Builds the verifier if needed (requires a Mac with the target Xcode/SDK, e.g. Xcode 27).
# Prints the raw verdict to stderr and the catalog line to stdout. Exit 0 iff runs=true.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="${1:?model-id required}"
BUNDLE="${2:?bundle dir/.aimodel required}"
KIND="${3:?kind required (llm|graph)}"
shift 3

INPUT=""; APPEND=""
while [ $# -gt 0 ]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --append) APPEND="$2"; shift 2 ;;
    *) shift ;;
  esac
done

BIN="$HERE/.build/release/coreai-runtime-verify"
if [ ! -x "$BIN" ]; then
  echo "[gate-c] building verifier…" >&2
  ( cd "$HERE" && swift build -c release >&2 )
fi

ARGS=(--model "$BUNDLE" --kind "$KIND")
[ -n "$INPUT" ] && ARGS+=(--input "$INPUT")

VERDICT="$("$BIN" "${ARGS[@]}")"
echo "$VERDICT" >&2

# Transform the verifier JSON into a catalog runtime-verifications.jsonl line.
LINE="$(printf '%s' "$VERDICT" | MODEL_ID="$MODEL_ID" python3 -c '
import json, os, sys
v = json.load(sys.stdin)
out = {
    "model_id": os.environ["MODEL_ID"],
    "kind": v.get("kind"),
    "loads": v.get("loads", False),
    "runs": v.get("runs", False),
    "runtime_os": v.get("runtime", {}).get("os", ""),
    "arch": v.get("runtime", {}).get("arch", ""),
    "output_preview": v.get("outputPreview"),
    "verified_at": v.get("verifiedAt"),
    "verifier": v.get("verifierVersion"),
    "source": "coreai-runtime-verify",
}
print(json.dumps(out))
')"

echo "$LINE"
if [ -n "$APPEND" ]; then
  echo "$LINE" >> "$APPEND"
  echo "[gate-c] appended to $APPEND" >&2
fi

printf '%s' "$VERDICT" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("runs") else 1)'
