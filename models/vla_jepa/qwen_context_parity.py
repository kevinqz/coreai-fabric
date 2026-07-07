# /// script
# requires-python = ">=3.12"
# ///
"""VLA-JEPA qwen_context parity.

This harness gates the exported qwen_context lane in models/vla_jepa/qwen_context.py:

  qwen_context(inputs_embeds, attention_mask, position_ids, embodied_positions)
    -> embodied_action_tokens

It compares the Torch export reference against the lowered Core AI function on
the same host-conditioned Qwen inputs. This is intentionally narrower than the
action-head parity harness: it checks the conditioned language lane only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen_context as qwen_context_export  # noqa: E402

REF_NPZ = "vla_jepa_qwen_context_parity_ref.npz"


def _bootstrap_ci(vals, n=1000, seed=0):
    a = np.asarray(vals, float)
    if len(a) < 2:
        return float(a.min()) if len(a) else 0.0, float(a.max()) if len(a) else 0.0
    g = np.random.default_rng(seed)
    mins = [g.choice(a, len(a), replace=True).min() for _ in range(n)]
    return float(np.percentile(mins, 2.5)), float(np.percentile(mins, 97.5))


def _build_reference(config_json: Path, instruction: str):
    model, sample, cfg, raw = qwen_context_export._build_sample(config_json, instruction)
    lang = qwen_context_export._build_language_inputs(model, sample)
    wrapper = qwen_context_export.QwenContextWrapper(model).eval()
    return cfg, raw, sample, lang, wrapper


def _run_torch(wrapper, lang):
    import torch

    with torch.no_grad():
        out = wrapper(
            lang["inputs_embeds"],
            lang["attention_mask"],
            lang["position_ids"],
            lang["embodied_positions"],
        )
    return out.float().numpy()


def cmd_reference(out: Path, config_json: Path, instruction: str, seed: int) -> None:
    cfg, raw, sample, lang, wrapper = _build_reference(config_json, instruction)
    ref = _run_torch(wrapper, lang)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / REF_NPZ,
        refs=np.asarray([ref], dtype=np.float32),
        inputs_embeds=np.asarray([lang["inputs_embeds"].float().numpy()], dtype=np.float32),
        attention_mask=np.asarray([lang["attention_mask"].numpy()], dtype=np.int64),
        position_ids=np.asarray([lang["position_ids"].numpy()], dtype=np.int64),
        embodied_positions=np.asarray([lang["embodied_positions"].numpy()], dtype=np.int64),
        hidden_size=int(ref.shape[-1]),
        seed=int(seed),
        image_keys=np.asarray(qwen_context_export._image_keys(raw), dtype=object),
        num_views=int(len(qwen_context_export._image_keys(raw))),
        num_embodied_tokens=int(cfg.num_embodied_action_tokens_per_instruction),
        instruction=instruction,
    )
    print(f"ok: wrote {out / REF_NPZ}")


def cmd_compare(out: Path, bundle: Path) -> None:
    import asyncio

    from coreai.runtime import AIModel, NDArray
    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _environment

    d = np.load(out / REF_NPZ, allow_pickle=True)
    refs = d["refs"]
    inputs_embeds = d["inputs_embeds"]
    attention_mask = d["attention_mask"]
    position_ids = d["position_ids"]
    embodied_positions = d["embodied_positions"]

    async def _run() -> dict:
        model = await AIModel.load(str(bundle))
        try:
            qwen = model.load_function("qwen_context")
        except Exception:  # noqa: BLE001
            return {
                "metric": "embodied_token_parity",
                "value": None,
                "status": "not_run",
                "reason": "asset lacks qwen_context entrypoint",
            }

        def nd(name, arr):
            dt = np.dtype(str(qwen.desc.input_descriptor(name).dtype))
            return NDArray(np.asarray(arr).astype(dt))

        outv = await qwen(
            inputs={
                "inputs_embeds": nd("inputs_embeds", inputs_embeds[0]),
                "attention_mask": nd("attention_mask", attention_mask[0]),
                "position_ids": nd("position_ids", position_ids[0]),
                "embodied_positions": nd("embodied_positions", embodied_positions[0]),
            }
        )
        a = outv["embodied_action_tokens"].numpy().astype(np.float64)
        r = refs[0].astype(np.float64)
        cos = float(np.dot(r.reshape(-1), a.reshape(-1)) / (np.linalg.norm(r) * np.linalg.norm(a) + 1e-12))
        lo, hi = _bootstrap_ci([cos])
        return {
            "metric": "embodied_token_parity",
            "value": cos,
            "status": "measured",
            "min_cosine": cos,
            "cosine_ci95": [lo, hi],
            "n_obs": 1,
            "hidden_size": int(r.shape[-1]),
            "num_tokens": int(r.shape[-2]),
            "runner": f"coreai-fabric-parity-runner/{__version__}",
            "environment": _environment(),
            "reference": "Torch qwen_context lane vs Core AI qwen_context over host-conditioned Qwen inputs.",
        }

    result = asyncio.run(_run())
    emit = out / "qwen-context-parity-measured.json"
    emit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"ok: wrote {emit}")


def main() -> None:
    ap = argparse.ArgumentParser(description="VLA-JEPA qwen_context parity")
    ap.add_argument("phase", nargs="?", default="reference", choices=["reference"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--config-json", type=Path)
    ap.add_argument("--instruction", default="Pick and place the object.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.compare:
        cmd_compare(args.out, args.bundle or (args.out / "qwen_context.aimodel"))
    else:
        if args.config_json is None:
            raise SystemExit("--config-json is required in reference mode")
        cmd_reference(args.out, args.config_json, args.instruction, args.seed)


if __name__ == "__main__":
    main()
