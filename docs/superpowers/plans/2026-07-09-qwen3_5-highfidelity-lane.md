# Qwen3.5 high-fidelity Core AI lane — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable `models/qwen3_5/` fabric lane that converts a Qwen3.5 hybrid VLM (pilot: `empero-ai/Qwythos-9B-Claude-Mythos-5-1M`) to a Core AI `.aimodel`, at high fidelity, with a full-provenance experiment ledger.

**Architecture:** Vendor the zoo's proven `qwen3_5.py` decode overlay (Ornith-9B / MiniCPM-V-4.6 ship on it) into fabric; add a thin config-derived `spec`, an experiment `ledger`, a two-venv parity harness (A/B/C/D + long-context margin rule), and the recipe/register/publish glue. Genuinely-new work (M-RoPE-on-hybrid, dynamic-res vision, 128K prefill, device matrix) is sequenced as gated spikes S1–S5 — each its own follow-on plan once its predecessor de-risks it.

**Tech Stack:** Python 3.12, PyTorch + `torch.export`, `coreai_torch.TorchConverter` (coremltools 9 underneath), `transformers` 5.x (two-venv reference), fabric CLI (`coreai-fabric`), pytest.

**Spec:** `docs/superpowers/specs/2026-07-09-qwen3_5-highfidelity-lane-design.md`.

**Grounding — real files this plan ports/uses (verified 2026-07-09):**
- Zoo overlay: `coreai-model-zoo/conversion/overlay/files/python/src/coreai_models/models/macos/qwen3_5.py` (`Qwen3_5GatedDeltaNet`, loop-free `_gated_delta_step`/`_gated_delta_chunk`, `Qwen3_5FullAttention`, `Qwen3_5DecodeCore`, `build_decode_state`, `DECODE_STATE_NAMES=("keyCache","valueCache","convState","recState")`, `qwen3_5_config_from_hf`), `+ qwen3_5_config.py`, `qwen3_5_gdn_metal.py`, `qwen3_5_metal_kernels.py`, and the iOS twins `models/ios/qwen3_5.py` / `qwen3_5_ios.py`.
- Zoo export scripts: `conversion/export_qwen3_5_decode_pipelined.py` (modes `fp16|int8|int8lin|int8hu|int4lin`, `--hf-id`, `--max-ctx`, `--head-sym`), `conversion/export_minicpmv46_vlm_pipelined.py`, `conversion/export_minicpmv46_vision.py`.
- Zoo gate: `_smoke/test_ornith9b_eager_gate.py`.
- Fabric precedents: `models/pi0fast/export.py` (tensor-I/O `TensorCache`), `models/vlm/parity.py` (two-venv A/B/C), `coreai_fabric/convert.py:194` (`is_script_tool`), `coreai_fabric/publish.py`, `coreai_fabric/register.py`, `schema/recipe.schema.json`.

---

## Scope & plan decomposition

This program is **six sequential phases**; later phases depend on earlier empirical outcomes (you cannot write faithful code for "dynamic-res vision export" before the port exists and the vision graph has been tried). Per the writing-plans scope rule, only the phases that are concretely code-specifiable **now** are written as full bite-sized TDD tasks here:

- **Phase 0 — Scaffold + experiment ledger** (fully detailed below). Working, tested software on its own.
- **Phase 1 — S1: port overlay + Qwythos decode gate + parity core** (fully detailed below). Ends with a numerically-gated Qwythos decode on the vendored overlay.
- **Phases 2–6 — S2 M-RoPE, S3 vision, S4 128K prefill, S5 device matrix, integration** (task-level outline at the end). Each becomes its own detailed plan authored **after** its predecessor's spike lands and writes its ledger entries.

Everything runs in the fabric repo (`coreai-fabric`), on a feature branch off `spec/qwen3_5-highfidelity-lane` (or a fresh worktree via `superpowers:using-git-worktrees`).

---

## Phase 0 — Scaffold + experiment ledger

### Task 0.1: Create the `models/qwen3_5/` package

**Files:**
- Create: `models/qwen3_5/__init__.py`
- Create: `models/qwen3_5/README.md`

- [ ] **Step 1: Create the package marker and readme**

`models/qwen3_5/__init__.py`:
```python
"""coreai-fabric Qwen3.5 high-fidelity Core AI lane.

Vendors the zoo's proven qwen3_5 decode overlay and adds fabric's spec,
experiment ledger, parity harness and recipe/register/publish glue.
See docs/superpowers/specs/2026-07-09-qwen3_5-highfidelity-lane-design.md.
"""
```

`models/qwen3_5/README.md`:
```markdown
# models/qwen3_5

Bespoke Qwen3.5 VLM exporter. Fabric does not run `.py` tools
(convert.py:194) — run `export.py` manually, drop the bundle at
`build/<id>/<id>.aimodel`, then `coreai-fabric verify <id>`.
Every optimization attempt is recorded via `ledger.py` (provenance/*.jsonl).
```

- [ ] **Step 2: Commit**

```bash
git add models/qwen3_5/__init__.py models/qwen3_5/README.md
git commit -m "feat(qwen3_5): scaffold the lane package"
```

### Task 0.2: Experiment ledger — data model

**Files:**
- Create: `models/qwen3_5/ledger.py`
- Test: `tests/test_qwen3_5_ledger.py`

- [ ] **Step 1: Write the failing test**

`tests/test_qwen3_5_ledger.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/kevinsaltarelli/Dev/Github/coreai-fabric && .venv/bin/python -m pytest tests/test_qwen3_5_ledger.py -v`
Expected: FAIL — `ModuleNotFoundError: models.qwen3_5.ledger`.

- [ ] **Step 3: Write minimal implementation**

`models/qwen3_5/ledger.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_qwen3_5_ledger.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add models/qwen3_5/ledger.py tests/test_qwen3_5_ledger.py
git commit -m "feat(qwen3_5): experiment ledger data model + validation"
```

### Task 0.3: Experiment ledger — append/round-trip

**Files:**
- Modify: `tests/test_qwen3_5_ledger.py`

- [ ] **Step 1: Write the failing test** (append to the test file)

```python
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
```

- [ ] **Step 2: Run to verify it passes** (impl already covers it)

Run: `.venv/bin/python -m pytest tests/test_qwen3_5_ledger.py -v`
Expected: PASS (5 passed). If `test_record_rejects_invalid_before_write` fails because the file was created, move the `LedgerEntry.from_dict(entry)` validation call in `record()` above the `mkdir`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_qwen3_5_ledger.py
git commit -m "test(qwen3_5): ledger append + validate-before-write"
```

---

## Phase 1 — S1: port the overlay + Qwythos decode gate + parity core

**Goal of S1:** reproduce Ornith-style eager numerics gate (fp16/int8lin/int8hu/int4lin) on **Qwythos-9B** using the vendored overlay, confirm the GVA 32v/16k branch + the 4-state layout, and stand up Gate D (state cosine). This is the highest-confidence phase — mostly a port.

### Task 1.1: Vendor the zoo overlay into the lane

**Files:**
- Create: `models/qwen3_5/overlay/macos/qwen3_5.py` (copy)
- Create: `models/qwen3_5/overlay/macos/qwen3_5_config.py`, `qwen3_5_gdn_metal.py`, `qwen3_5_metal_kernels.py` (copies)
- Create: `models/qwen3_5/overlay/ios/qwen3_5.py`, `qwen3_5_ios.py` (copies, for the device matrix in S5)
- Create: `models/qwen3_5/overlay/PROVENANCE.md`

- [ ] **Step 1: Copy the overlay verbatim, recording provenance**

```bash
ZOO=/Users/kevinsaltarelli/Dev/Github/coreai-model-zoo
OVL=$ZOO/conversion/overlay/files/python/src/coreai_models/models
mkdir -p models/qwen3_5/overlay/macos models/qwen3_5/overlay/ios
cp $OVL/macos/qwen3_5.py $OVL/macos/qwen3_5_config.py $OVL/macos/qwen3_5_gdn_metal.py $OVL/macos/qwen3_5_metal_kernels.py models/qwen3_5/overlay/macos/
cp $OVL/ios/qwen3_5.py $OVL/ios/qwen3_5_ios.py models/qwen3_5/overlay/ios/
( cd $ZOO && git rev-parse HEAD ) > models/qwen3_5/overlay/PROVENANCE.md
```

Prepend to `PROVENANCE.md` (edit): the source repo, the paths copied, and "vendored verbatim; local edits tracked below" — so the port stays auditable.

- [ ] **Step 2: Verify the overlay imports its public API**

Run:
```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'models/qwen3_5/overlay/macos'); import qwen3_5 as q; print([n for n in ('Qwen3_5GatedDeltaNet','Qwen3_5FullAttention','Qwen3_5DecodeCore','build_decode_state','DECODE_STATE_NAMES','qwen3_5_config_from_hf') if hasattr(q,n)])"
```
Expected: all six names printed. If import fails on `from coreai_torch.composite_ops import GatedDeltaUpdate`, that dependency must be present in `.venv` — note it as an env prerequisite in `PROVENANCE.md` and install the pinned `coreai_torch`.

- [ ] **Step 3: Commit**

```bash
git add models/qwen3_5/overlay/
git commit -m "feat(qwen3_5): vendor zoo qwen3_5 overlay (macos+ios) with provenance"
```

### Task 1.2: `spec.py` — typed spec from a HF config

**Files:**
- Create: `models/qwen3_5/spec.py`
- Test: `tests/test_qwen3_5_spec.py`
- Fixture: `tests/fixtures/qwythos_text_config.json` (the `text_config` + `vision_config` block from the pilot config, saved offline so the test is network-free)

- [ ] **Step 1: Save the fixture** (the verified pilot values)

`tests/fixtures/qwythos_text_config.json`:
```json
{"text_config":{"model_type":"qwen3_5_text","hidden_size":4096,"num_hidden_layers":32,
"num_attention_heads":16,"num_key_value_heads":4,"head_dim":256,"intermediate_size":12288,
"vocab_size":248320,"rms_norm_eps":1e-06,"partial_rotary_factor":0.25,"attn_output_gate":true,
"full_attention_interval":4,"linear_conv_kernel_dim":4,"linear_key_head_dim":128,
"linear_num_key_heads":16,"linear_num_value_heads":32,"linear_value_head_dim":128,
"mtp_num_hidden_layers":1,"max_position_embeddings":1048576,
"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144,
"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_theta":10000000}},
"vision_config":{"model_type":"qwen3_5_vision","depth":27,"hidden_size":1152,"num_heads":16,
"patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2,"out_hidden_size":4096,
"num_position_embeddings":2304,"deepstack_visual_indexes":[]},
"image_token_id":248056,"vision_start_token_id":248053}
```

- [ ] **Step 2: Write the failing test**

`tests/test_qwen3_5_spec.py`:
```python
import json
from pathlib import Path
from models.qwen3_5.spec import load_spec

FIX = Path(__file__).parent / "fixtures" / "qwythos_text_config.json"


def test_layer_types_derived():
    s = load_spec(json.loads(FIX.read_text()))
    assert s.num_layers == 32
    assert s.layer_types.count("full") == 8
    assert s.layer_types.count("linear") == 24
    assert s.layer_types[:4] == ["linear", "linear", "linear", "full"]


def test_dims_and_derived():
    s = load_spec(json.loads(FIX.read_text()))
    assert s.conv_dim == 8192            # key_dim*2 + value_dim = 2048*2 + 4096
    assert s.rotary_dim == 64            # partial_rotary_factor 0.25 * head_dim 256
    assert s.recstate_shape == (32, 128, 128)   # num_v, head_k, head_v
    assert s.image_token_id == 248056
    assert s.yarn["original_max_position_embeddings"] == 262144
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_qwen3_5_spec.py -v`
Expected: FAIL — `ModuleNotFoundError: models.qwen3_5.spec`.

- [ ] **Step 4: Write minimal implementation**

`models/qwen3_5/spec.py`:
```python
"""Typed Qwen3.5 spec derived from a HF config dict. Single source of truth."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen35Spec:
    num_layers: int
    layer_types: list[str]
    hidden_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    conv_dim: int
    conv_kernel: int
    rotary_dim: int
    recstate_shape: tuple[int, int, int]
    image_token_id: int
    vision_start_token_id: int
    yarn: dict
    mrope_section: list[int]
    vision: dict


def load_spec(config: dict) -> Qwen35Spec:
    tc = config["text_config"]
    interval = tc["full_attention_interval"]
    layer_types = [
        "full" if (i + 1) % interval == 0 else "linear"
        for i in range(tc["num_hidden_layers"])
    ]
    key_dim = tc["linear_key_head_dim"] * tc["linear_num_key_heads"]
    value_dim = tc["linear_value_head_dim"] * tc["linear_num_value_heads"]
    rp = tc["rope_parameters"]
    return Qwen35Spec(
        num_layers=tc["num_hidden_layers"],
        layer_types=layer_types,
        hidden_size=tc["hidden_size"],
        head_dim=tc["head_dim"],
        num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"],
        conv_dim=key_dim * 2 + value_dim,
        conv_kernel=tc["linear_conv_kernel_dim"],
        rotary_dim=int(tc["head_dim"] * tc["partial_rotary_factor"]),
        recstate_shape=(tc["linear_num_value_heads"], tc["linear_key_head_dim"], tc["linear_value_head_dim"]),
        image_token_id=config["image_token_id"],
        vision_start_token_id=config["vision_start_token_id"],
        yarn=rp,
        mrope_section=rp["mrope_section"],
        vision=config["vision_config"],
    )
```

Note: `layer_types` here is derived from `full_attention_interval`; assert in Step 5 it matches the config's own `layer_types` array when present.

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_qwen3_5_spec.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add models/qwen3_5/spec.py tests/test_qwen3_5_spec.py tests/fixtures/qwythos_text_config.json
git commit -m "feat(qwen3_5): typed spec from HF config"
```

### Task 1.3: Reproduce the eager numerics gate on Qwythos

**Files:**
- Create: `models/qwen3_5/gate.py` (thin wrapper adapting the zoo `_smoke/test_ornith9b_eager_gate.py` to the vendored overlay + `--hf-id`)
- Create: `models/qwen3_5/provenance/` (dir; ledger output lands here)

- [ ] **Step 1: Port the gate**

Adapt `coreai-model-zoo/_smoke/test_ornith9b_eager_gate.py` into `models/qwen3_5/gate.py`: point `load_export_module()` at `models/qwen3_5/overlay/macos/qwen3_5.py`, keep its per-layer 24/24 gate logic, add a `--hf-id` default of `empero-ai/Qwythos-9B-Claude-Mythos-5-1M` and an `--exp` id, and on completion call `models.qwen3_5.ledger.record(...)` with the gate results into `models/qwen3_5/provenance/s1.jsonl`.

- [ ] **Step 2: Run the gate (weights required; the real S1 deliverable)**

Run (in the reference venv with transformers pinned):
```bash
.venv-lerobot/bin/python models/qwen3_5/gate.py --hf-id empero-ai/Qwythos-9B-Claude-Mythos-5-1M --mode int8lin --exp s1-int8lin
```
Expected: per-layer gate `24/24` (matching Ornith-9B). **Gate:** if any layer < exact tolerance, STOP — record the failure + which layer/mechanism in the ledger before proceeding (this is where the missing-mechanism risks — qk-norm, output gate — would surface).

- [ ] **Step 3: Confirm the GVA branch + 4-state layout**

Run:
```bash
.venv-lerobot/bin/python -c "import sys; sys.path.insert(0,'models/qwen3_5/overlay/macos'); import qwen3_5 as q; print(q.DECODE_STATE_NAMES); import json; from models.qwen3_5.spec import load_spec; s=load_spec(json.load(open('tests/fixtures/qwythos_text_config.json'))); print('recstate',s.recstate_shape,'conv_dim',s.conv_dim)"
```
Expected: `('keyCache','valueCache','convState','recState')` and `recstate (32,128,128) conv_dim 8192` — confirming 32v/16k GVA + the 2-extra-state layout.

- [ ] **Step 4: Commit**

```bash
git add models/qwen3_5/gate.py models/qwen3_5/provenance/
git commit -m "feat(qwen3_5): Qwythos eager numerics gate + S1 ledger"
```

### Task 1.4: Parity harness core — Gate D (state cosine)

**Files:**
- Create: `models/qwen3_5/parity.py` (two-venv skeleton adapted from `models/vlm/parity.py`, with M-RoPE re-enabled — do NOT copy its `rope_scaling=None` downgrade)
- Test: `tests/test_qwen3_5_parity_gated.py` (pure-function tests for the metrics; the full two-venv run is a manual command)

- [ ] **Step 1: Write the failing metric test**

`tests/test_qwen3_5_parity_gated.py`:
```python
import torch
from models.qwen3_5.parity import state_cosine, token_match_rate


def test_state_cosine_identical_is_one():
    a = torch.randn(24, 32, 128, 128)
    assert state_cosine(a, a.clone()) > 0.9999


def test_state_cosine_detects_drift():
    a = torch.randn(24, 32, 128, 128)
    b = a + 0.5 * torch.randn_like(a)
    assert state_cosine(a, b) < 0.99


def test_token_match_rate():
    ref = [1, 2, 3, 4, 5]
    got = [1, 2, 9, 4, 5]
    assert token_match_rate(ref, got) == 0.8
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_qwen3_5_parity_gated.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation** (metrics only; the two-venv driver is added around them)

`models/qwen3_5/parity.py` (metrics section):
```python
"""Two-venv greedy parity for the Qwen3.5 lane: Gate A token-exact, B first-token
logit cosine, C vision-feature cosine, D recurrent-state cosine. Long context uses
a margin rule + rollout sanity, not bit-exact >=0.999 (fp16 noise compounds).
M-RoPE is kept (unlike models/vlm/parity.py, which strips it)."""
from __future__ import annotations
import torch


def state_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    fa, fb = a.flatten().float(), b.flatten().float()
    return float(torch.nn.functional.cosine_similarity(fa, fb, dim=0))


def token_match_rate(ref: list[int], got: list[int]) -> float:
    n = min(len(ref), len(got))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if ref[i] == got[i]) / n
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_qwen3_5_parity_gated.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add models/qwen3_5/parity.py tests/test_qwen3_5_parity_gated.py
git commit -m "feat(qwen3_5): parity metrics (Gate A token-exact, Gate D state cosine)"
```

**End of Phase 1.** Deliverable: Qwythos passes the eager gate on the vendored overlay, the 4-state layout + GVA branch are confirmed, and the parity metrics exist. S1 ledger entries recorded. This de-risks the port and unblocks writing the Phase 2 plan.

---

## Phases 2–6 — outline (each becomes its own detailed plan after its predecessor lands)

Written at task granularity now; expand to full bite-sized TDD once the prior spike's ledger confirms the approach.

**Phase 2 — S2: M-RoPE on the hybrid backbone.** Re-enable interleaved 3-D M-RoPE + YaRN `attention_scaling` mscale in the vendored `Qwen3_5FullAttention.apply_rope`; add `tests/test_qwen3_5_mrope.py` asserting `mrope_section [11,11,10]` axis split + partial `rotary_dim 64` (`q_pass` untouched) against the transformers reference `Qwen3_5TextRotaryEmbedding`. Gate: first-token logit cosine vs torch ≥ 0.999. Ledger the delta vs the 1-D-collapsed baseline.

**Phase 3 — S3: dynamic-res vision (primary unknown).** Port `export_minicpmv46_vision.py` as the starting shape; implement `models/qwen3_5/vision.py` with Conv3d patch embed, 2-D vision RoPE, the learned `Embedding(2304,1152)` bilinear interpolation, cu_seqlens full attention (no windowing), and the merger. **Attempt native dynamic-res first** (`torch.export` with dynamic shapes → `coreai_torch`); if it won't lower, document the exact failure and implement each exporting alternative (resolution-bucketing over a grid set; fixed-max slicing), each shipped + ledgered with its Gate-C vision-feature-cosine delta.

**Phase 4 — S4: 128K loop-free prefill.** Use the overlay's `_gated_delta_chunk` (loop-free) — NOT the `GatedDeltaUpdate` `scf.while` composite (does not lower on device). Compare in-graph chunk (fp16 safe ≤ 8) vs the `qwen3_5_gdn_metal.py` fp32 Metal kernel (GPU-only, ≤ 64). Deliverable: 128K export (`--max-ctx 131072`) + a TTFT number + the margin-rule + rollout-sanity long-context gate. Ledger each prefill variant.

**Phase 5 — S5: device matrix + quant.** From the one flow, emit every feasible variant: Mac (`export_qwen3_5_decode_pipelined.py` modes int8lin/int8hu/int4lin, `--head-sym`) and iPhone (the vendored `overlay/ios/` twins, int4-body + int8-head where ≤ ~6.4 GB). Each variant: eager gate + engine token-exact + bundle size + tok/s, all ledgered. Do not assume int4 fails (Ornith passes 24/24).

**Phase 6 — Integration: recipe + register + publish + catalog.** Author `recipes/qwen3_5-qwythos-9b.yaml` with the schema-required top-level keys `[id, upstream, conversion, expected, parity, publish, status]` (+ `conversion.{tool,quantization,precision}`, `upstream.{hf_repo,license: apache-2.0, license_terms: permissive}`, `parity.{gate_a,gate_b}`, `publish.{hf_target_namespace: kevinqz, repo_name}`, a full `catalog:` block). Run `coreai-fabric verify qwen3_5-qwythos-9b`, then `coreai-fabric register --catalog-path <clone>`. Catalog entry: `source_group: fabric` + artifact `group: external` + `officiality.apple_export_recipe: false`; `architecture: transformer` (hybrid detail in `notes`); `capabilities: [vision-language, hybrid-llm, reasoning, agentic]`; `bundle_kind: vlm`; NO `catalog.traits`, NO `framework_contract`; new id `qwythos-9b`; add a verified `empero-ai/Qwythos` `original_model_sources` upstream with `license_terms: permissive` (laundering guard); `context_window: 128K`, `streaming: true`. Publish gate: apache-2.0 passes; consider a sharded upload for the ~10–18 GB bundle (R6).

---

## Self-Review

**Spec coverage:** §5 modules → Phase 0 (ledger) + Phase 1 (spec, overlay, parity) + Phases 2–6 (vision, export, recipe/register/publish); §6/§7 cores + state → Task 1.1/1.3 (ported) + S2/S4; §8 vision → S3; §10 ledger → Phase 0; §11 parity → Task 1.4 + gates in each spike; §12 spikes S1–S5 → Phases 1–5; §13 integration → Phase 6; device matrix (§1) → S5. All spec sections map to a task or phase.

**Placeholder scan:** Phase 0 + Phase 1 contain complete code/commands. Phases 2–6 are intentionally task-level (their code depends on empirical spike outcomes) — flagged as "expand to a full plan after the predecessor lands," which is the honest decomposition, not a hidden TODO.

**Type consistency:** `LedgerEntry`/`record`/`REQUIRED_FIELDS` (ledger), `Qwen35Spec`/`load_spec` (spec), `state_cosine`/`token_match_rate` (parity), `DECODE_STATE_NAMES` (overlay) are used consistently across tasks and match the real overlay symbols.
