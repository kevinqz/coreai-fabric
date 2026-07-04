#!/usr/bin/env bash
# Set up venv-A — the LeRobot export environment for the VLA/action lane.
#
# WHY A SECOND VENV (the "two-venv dance"):
#   lerobot[pi] 0.5.1 pins transformers==5.3.0 + numpy<2.3 + huggingface-hub>=1.3,
#   which conflict HARD with the fabric .venv (venv-B: transformers 4.57.6,
#   numpy 2.3.5, coreai-core's hub 0.x). They CANNOT share a venv.
#   BUT torch is NOT the conflict: lerobot pins torch>=2.7,<2.11, so we pin BOTH
#   venvs to torch==2.9.0. That means a torch.export ExportedProgram serialized
#   here (venv-A) loads unchanged in venv-B — no cross-version .pt2 risk.
#
# The export step (torch.export) runs HERE (venv-A). The lower step
# (coreai_torch.TorchConverter) runs in the fabric .venv (venv-B) on the
# serialized .pt2. See docs/vla-export-runbook.md.
#
# DISK: lerobot[pi] pulls a large dep tree (~4-6GB incl. torch). Only run this
# when the machine has real headroom (see the runbook's disk gate).
set -euo pipefail

VENV="${1:-/Users/kevinsaltarelli/Dev/Github/coreai-fabric/.venv-lerobot}"
PY="${PYTHON:-python3.12}"   # lerobot requires-python >= 3.12 (VERIFIED); 3.13 also OK

echo "==> creating venv-A at $VENV (python: $PY)"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip

# Pin torch FIRST to 2.9.0 (== venv-B) so the resolver can't drag in 2.10 —
# this ordering is load-bearing: it keeps the ExportedProgram same-version.
pip install "torch==2.9.0" "torchvision==0.24.0"
# The [pi] extra is what pulls transformers==5.3.0 + scipy + numpy<2.3.
# Bare `lerobot` leaves transformers ABSENT -> PI0Policy import fails
# (CONFIG_MAPPING is None); the latest transformers breaks lerobot's groot
# import. The [pi] pin is the fix for both.
pip install "lerobot[pi]==0.5.1"
pip check || echo "WARN: pip check flagged conflicts — if it's torchvision<->torch, re-run: pip install torch==2.9.0 torchvision (unpinned) and read back the resolved version"

# Reproducibility gate — hard-fail if the resolver drifted (never a silent 2.10).
python - <<'PY'
import torch, transformers, numpy
assert torch.__version__.startswith("2.9"), f"torch drifted: {torch.__version__} (must be 2.9.x == venv-B)"
assert transformers.__version__.startswith("5."), f"transformers: {transformers.__version__} (expected 5.3.0)"
assert numpy.__version__ < "2.3", f"numpy: {numpy.__version__} (must be <2.3 for lerobot)"
from lerobot.policies.pi0.modeling_pi0 import PI0Config, PI0Policy, PI0Pytorch  # import gate
print("venv-A OK:", "torch", torch.__version__, "| transformers", transformers.__version__, "| numpy", numpy.__version__)
PY
echo "==> venv-A ready. Next: scripts/pi0_export_probe.py (see docs/vla-export-runbook.md)"
