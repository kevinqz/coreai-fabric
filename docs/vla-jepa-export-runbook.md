# VLA-JEPA export runbook

VLA-JEPA is a LeRobot VLA policy that combines Qwen3-VL-2B-Instruct, V-JEPA2,
and a flow-matching DiT action head. The LeRobot docs are explicit about the
runtime split: the JEPA world model is training-only; inference uses Qwen plus
the action head.

## Current upstream checkpoints

- `lerobot/VLA-JEPA-LIBERO`: LIBERO-10, 2 cameras, state dim 8, action dim 7.
- `lerobot/VLA-JEPA-Pretrain`: DROID pretrain, 2 exterior-left cameras, action dim 7.
- `lerobot/VLA-JEPA-SimplerEnv`: Bridge/RT-1 SimplerEnv, 1 camera, action dim 7.

All three recipes are draft-only until a real export and `action_parity`
measurement have been run.

## Phase 0 — install the policy source

The current `.venv-lerobot` in this workspace has LeRobot installed, but not the
new `lerobot.policies.vla_jepa` module. The HF docs page for VLA-JEPA is on the
`main` docs stream and notes that main requires installation from source.

Use the source install before any export attempt:

```bash
.venv-lerobot/bin/python -m pip install -U "git+https://github.com/huggingface/lerobot.git"
.venv-lerobot/bin/python - <<'PY'
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy
print(VLAJEPAPolicy.name)
PY
```

## Phase 1 — prove the action-head lane first

Do not start by tracing all of Qwen3-VL. First load the policy and isolate the
flow-matching action head. The target export contract is:

- `qwen_context`: host/tokenizer/Qwen path produces embodied action tokens.
- `action_denoise_step`: Core AI graph maps `(embodied_action_tokens, x_t,
  timestep, state?) -> velocity`.

Gate this with a fixed-noise action parity harness before adding the full Qwen
context path. The JEPA predictor is not part of inference and must not be needed
for the runtime asset.

Fast op-coverage probe:

```bash
.venv-lerobot/bin/python models/vla_jepa/export.py export-action-head \
  --config-json build/_vla_jepa/VLA-JEPA-LIBERO/config.json \
  --out build/vla-jepa-libero --probe-small
.venv/bin/python models/vla_jepa/export.py --lower --out build/vla-jepa-libero
```

DONE (2026-07-06, macOS 27): both the tiny `--probe-small` graph and the real
DiT-B action-head dimensions exported and lowered successfully through
coreai-torch 0.4.1. Real-dimension LIBERO action-head probe:
`conditioning_tokens=[1,32,2048]`, `x_t=[1,7,7]`, `timestep=[1]`,
`state=[1,1,8]` -> `velocity=[1,7,7]`; produced
`action_denoise_step.pt2` (~608 MB) and `vla-jepa-libero.aimodel` (~589 MB).
`coreai-fabric verify vla-jepa-libero` reports Gate A passed and Gate B
`not_run`, as expected until the fixed-noise action_parity harness is wired.

Full action-head export uses the real DiT-B dimensions and, when the upstream
`model.safetensors` is present locally, streams only `model.action_model.*`
from the checkpoint:

```bash
.venv-lerobot/bin/python models/vla_jepa/export.py export-action-head \
  --config-json build/_vla_jepa/VLA-JEPA-LIBERO/config.json \
  --weights build/_vla_jepa/VLA-JEPA-LIBERO/model.safetensors \
  --out build/vla-jepa-libero
.venv/bin/python models/vla_jepa/export.py --lower --out build/vla-jepa-libero
```

Because LeRobot places `model.action_model.*` first in the safetensors file, the
current lane only needs the first `310580742` bytes of
`VLA-JEPA-LIBERO/model.safetensors` (header + contiguous action-head payload).
That means the real action-head export can be staged with a byte-range fetch
plus subset extraction instead of waiting for the full ~6.1 GB checkpoint:

```bash
curl -L -r 0-310580741 \
  -o build/_vla_jepa/VLA-JEPA-LIBERO/model.safetensors \
  https://huggingface.co/lerobot/VLA-JEPA-LIBERO/resolve/735d9f692981e286ade093b5046627eda876e5d0/model.safetensors
python models/vla_jepa/extract_action_head.py \
  --src build/_vla_jepa/VLA-JEPA-LIBERO/model.safetensors \
  --out build/_vla_jepa/VLA-JEPA-LIBERO/action_model.safetensors
.venv-lerobot/bin/python models/vla_jepa/export.py export-action-head \
  --config-json build/_vla_jepa/VLA-JEPA-LIBERO/config.json \
  --weights build/_vla_jepa/VLA-JEPA-LIBERO/action_model.safetensors \
  --out build/vla-jepa-libero
```

Action-head Gate B follows the same two-venv discipline as the other VLA
harnesses. It compares the Torch action head against Core AI
`action_denoise_step` over identical synthetic Qwen-context tokens, fixed noise,
state, and the 4-step VLA-JEPA Euler loop:

```bash
.venv-lerobot/bin/python models/vla_jepa/parity.py reference \
  --config-json build/_vla_jepa/VLA-JEPA-LIBERO/config.json \
  --weights build/_vla_jepa/VLA-JEPA-LIBERO/model.safetensors \
  --out build/vla-jepa-libero
.venv/bin/python models/vla_jepa/parity.py --compare --out build/vla-jepa-libero
coreai-fabric verify vla-jepa-libero
```

## Phase 2 — Qwen context path

Once the action head lowers, export or reuse the same Qwen3-VL context path used
by the VLM lane. The parity harness must feed the same images, instruction, state
and fixed initial noise through both torch and the asset, then compare the final
normalized action chunk after the 4-step Euler loop.

## Phase 3 — publish discipline

Publish only after:

- Gate A passes on the actual bundle inventory.
- Gate B records measured `action_parity` in `action-parity-measured.json`.
- The card labels the result as conversion fidelity, not task success.

The upstream model cards currently say no policy evaluation results are provided
there; do not add task-success claims from conversion parity.
