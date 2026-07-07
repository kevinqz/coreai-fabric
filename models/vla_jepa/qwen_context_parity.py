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


def _build_reference_model(config_json: Path):
    prep = qwen_context_export._prepare_model(config_json)
    qwen_model = prep[0]
    wrapper = qwen_context_export.QwenContextWrapper(qwen_model).eval()
    return (*prep, wrapper)


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


def cmd_reference(out: Path, config_json: Path, instruction: str, n_obs: int, seed: int) -> None:
    (
        qwen_model,
        processor,
        cfg,
        raw,
        action_prompt,
        embodied_prompt,
        embodied_action_token_id,
        wrapper,
    ) = _build_reference_model(config_json)

    refs, ie_list, am_list, pos_list, emb_list = [], [], [], [], []
    base_shape = None
    for i in range(n_obs):
        # Distinct seeded synthetic images per observation; fixed instruction +
        # image grid keep the exported graph's static shapes identical, so only
        # inputs_embeds VALUES vary. Observation 0 uses the canonical blank
        # (zero) images used at export time.
        images = qwen_context_export._make_images(cfg, raw, seed=None if i == 0 else seed + i)
        sample = qwen_context_export._sample_from_images(
            processor, cfg, images, instruction, action_prompt, embodied_prompt, embodied_action_token_id
        )
        lang = qwen_context_export._build_language_inputs(qwen_model, sample)
        shp = tuple(lang["inputs_embeds"].shape)
        if base_shape is None:
            base_shape = shp
        elif shp != base_shape:
            raise SystemExit(
                f"obs {i} inputs_embeds shape {shp} != obs 0 {base_shape}; the static qwen_context "
                "asset requires a fixed shape. Keep the instruction and image grid constant across "
                "observations (vary only the image pixels)."
            )
        refs.append(_run_torch(wrapper, lang))
        ie_list.append(lang["inputs_embeds"].float().numpy())
        am_list.append(lang["attention_mask"].numpy())
        pos_list.append(lang["position_ids"].numpy())
        emb_list.append(lang["embodied_positions"].numpy())

    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / REF_NPZ,
        refs=np.stack(refs).astype(np.float32),
        inputs_embeds=np.stack(ie_list).astype(np.float32),
        attention_mask=np.stack(am_list).astype(np.int64),
        position_ids=np.stack(pos_list).astype(np.int64),
        embodied_positions=np.stack(emb_list).astype(np.int64),
        hidden_size=int(refs[0].shape[-1]),
        n_obs=int(n_obs),
        seed=int(seed),
        image_keys=np.asarray(qwen_context_export._image_keys(raw), dtype=object),
        num_views=int(len(qwen_context_export._image_keys(raw))),
        num_embodied_tokens=int(cfg.num_embodied_action_tokens_per_instruction),
        instruction=instruction,
    )
    print(f"ok: wrote {out / REF_NPZ} (n_obs={n_obs})")


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

        cosines = []
        for i in range(len(refs)):
            outv = await qwen(
                inputs={
                    "inputs_embeds": nd("inputs_embeds", inputs_embeds[i]),
                    "attention_mask": nd("attention_mask", attention_mask[i]),
                    "position_ids": nd("position_ids", position_ids[i]),
                    "embodied_positions": nd("embodied_positions", embodied_positions[i]),
                }
            )
            a = outv["embodied_action_tokens"].numpy().astype(np.float64)
            r = refs[i].astype(np.float64)
            cosines.append(
                float(np.dot(r.reshape(-1), a.reshape(-1)) / (np.linalg.norm(r) * np.linalg.norm(a) + 1e-12))
            )
        cos = np.asarray(cosines)
        lo, hi = _bootstrap_ci(cos.tolist())
        return {
            "metric": "embodied_token_parity",
            "value": float(cos.min()),
            "status": "measured",
            "min_cosine": float(cos.min()),
            "median_cosine": float(np.median(cos)),
            "mean_cosine": float(cos.mean()),
            "cosine_ci95": [lo, hi],
            "per_obs_cosine": [float(x) for x in cosines],
            "n_obs": int(len(refs)),
            "hidden_size": int(refs.shape[-1]),
            "num_tokens": int(refs.shape[-2]),
            "runner": f"coreai-fabric-parity-runner/{__version__}",
            "environment": _environment(),
            "reference": (
                "Torch qwen_context lane vs Core AI qwen_context over host-conditioned Qwen inputs; "
                "distinct seeded synthetic images per observation, fixed instruction and image grid."
            ),
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
    ap.add_argument("--n-obs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.compare:
        cmd_compare(args.out, args.bundle or (args.out / "qwen_context.aimodel"))
    else:
        if args.config_json is None:
            raise SystemExit("--config-json is required in reference mode")
        cmd_reference(args.out, args.config_json, args.instruction, args.n_obs, args.seed)


if __name__ == "__main__":
    main()
