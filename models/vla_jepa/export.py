# /// script
# requires-python = ">=3.12"
# ///
"""VLA-JEPA action-head export lane.

VLA-JEPA inference is Qwen3-VL context tokens + a flow-matching DiT action head.
The JEPA world model is training-only. This script deliberately starts with the
smallest faithful exportable unit:

  action_denoise_step(conditioning_tokens, x_t, timestep[, state]) -> velocity

The host owns Qwen/tokenization/image preprocessing and the 4-step Euler loop.
That mirrors the pi0/smolvla split-export discipline: export the deterministic
step graph first, then prove Gate B with fixed-noise action_parity before
claiming a complete policy.

Usage:
  .venv-lerobot/bin/python models/vla_jepa/export.py export-action-head \
      --config-json build/_vla_jepa/VLA-JEPA-LIBERO/config.json \
      --out build/vla-jepa-libero --probe-small
  .venv/bin/python models/vla_jepa/export.py --lower --out build/vla-jepa-libero
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

QWEN3_VL_2B_HIDDEN = 2048


def _load_action_config(config_json: Path | None, probe_small: bool):
    from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig

    cfg = VLAJEPAConfig()
    raw = json.loads(config_json.read_text()) if config_json else {}
    valid = {f.name for f in dataclasses.fields(VLAJEPAConfig)}
    for key, value in raw.items():
        if key in valid and not isinstance(value, (dict, list)):
            setattr(cfg, key, value)
    if probe_small:
        cfg.action_model_type = "DiT-test"
        cfg.action_hidden_size = 16
        cfg.action_num_layers = 2
        cfg.action_num_heads = 2
        cfg.action_attention_head_dim = 8
        cfg.num_embodied_action_tokens_per_instruction = 4
        cfg.action_max_seq_len = 64
    return cfg, raw


def _has_state(raw: dict) -> bool:
    return any(v.get("type") == "STATE" for v in (raw.get("input_features") or {}).values())


class DenoiseStepWrapper:
    @staticmethod
    def build(head, has_state: bool):
        import torch

        class _W(torch.nn.Module):
            def __init__(self, h):
                super().__init__()
                self.h = h

            def forward(self, conditioning_tokens, x_t, timestep, state=None):
                hidden = self.h._build_inputs(conditioning_tokens, x_t, state, timestep)
                pred = self.h.model(
                    hidden_states=hidden,
                    encoder_hidden_states=conditioning_tokens,
                    timestep=timestep,
                )
                return self.h.action_decoder(pred[:, -self.h.action_horizon :])

        class _NoState(torch.nn.Module):
            def __init__(self, h):
                super().__init__()
                self.h = h

            def forward(self, conditioning_tokens, x_t, timestep):
                hidden = self.h._build_inputs(conditioning_tokens, x_t, None, timestep)
                pred = self.h.model(
                    hidden_states=hidden,
                    encoder_hidden_states=conditioning_tokens,
                    timestep=timestep,
                )
                return self.h.action_decoder(pred[:, -self.h.action_horizon :])

        return (_W if has_state else _NoState)(head).eval()


def _load_action_weights(head, weights: Path | None) -> None:
    if weights is None:
        return
    from safetensors.torch import load_file

    raw = load_file(str(weights), device="cpu")
    prefix = "model.action_model."
    subset = {k[len(prefix) :]: v for k, v in raw.items() if k.startswith(prefix)}
    missing, unexpected = head.load_state_dict(subset, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected action-head keys: {unexpected[:8]}")
    if missing:
        print(f"warning: missing {len(missing)} action-head key(s); first: {missing[:8]}")


def cmd_export_action_head(args) -> None:
    import torch
    from lerobot.policies.vla_jepa.action_head import VLAJEPAActionHead

    cfg, raw = _load_action_config(args.config_json, args.probe_small)
    has_state = args.with_state if args.with_state is not None else _has_state(raw)
    head = VLAJEPAActionHead(cfg, cross_attention_dim=args.cross_attention_dim).to("cpu").eval()
    _load_action_weights(head, args.weights)
    wrapper = DenoiseStepWrapper.build(head, has_state)

    b = 1
    cond_tokens = int(cfg.num_embodied_action_tokens_per_instruction)
    conditioning = torch.zeros(b, cond_tokens, args.cross_attention_dim)
    x_t = torch.zeros(b, int(cfg.chunk_size), int(cfg.action_dim))
    timestep = torch.zeros(b, dtype=torch.long)
    state = torch.zeros(b, 1, int(cfg.state_dim))
    export_args = (conditioning, x_t, timestep, state) if has_state else (conditioning, x_t, timestep)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        ep = torch.export.export(wrapper, args=export_args, strict=False)
    torch.export.save(ep, str(out / "action_denoise_step.pt2"))
    contract = {
        "entrypoint": "action_denoise_step",
        "has_state": has_state,
        "conditioning_tokens": cond_tokens,
        "cross_attention_dim": args.cross_attention_dim,
        "chunk_size": int(cfg.chunk_size),
        "action_dim": int(cfg.action_dim),
        "state_dim": int(cfg.state_dim),
        "num_inference_timesteps": int(cfg.num_inference_timesteps),
        "probe_small": bool(args.probe_small),
    }
    (out / "vla-jepa-action-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"ok: wrote {out}/action_denoise_step.pt2")
    print(f"ok: wrote {out}/vla-jepa-action-contract.json")


def cmd_lower(args) -> None:
    import torch
    from coreai_torch import TorchConverter, get_decomp_table

    out = args.out
    contract = json.loads((out / "vla-jepa-action-contract.json").read_text())
    input_names = ["conditioning_tokens", "x_t", "timestep"]
    if contract["has_state"]:
        input_names.append("state")
    ep = torch.export.load(str(out / "action_denoise_step.pt2"))
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(
        ep,
        input_names=input_names,
        output_names=["velocity"],
        entrypoint_name="action_denoise_step",
    )
    prog = conv.to_coreai()
    prog.optimize()
    aimodel = out / f"{out.name}.aimodel"
    prog.save_asset(aimodel)
    print(f"ok: lowered VLA-JEPA action head -> {aimodel}")


def main() -> None:
    ap = argparse.ArgumentParser(description="VLA-JEPA action-head export")
    ap.add_argument("phase", nargs="?", default="export-action-head", choices=["export-action-head"])
    ap.add_argument("--lower", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config-json", type=Path)
    ap.add_argument("--weights", type=Path, help="local upstream model.safetensors; loads model.action_model.*")
    ap.add_argument("--cross-attention-dim", type=int, default=QWEN3_VL_2B_HIDDEN)
    ap.add_argument("--with-state", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--probe-small", action="store_true", help="use tiny DiT-test dimensions for op coverage")
    args = ap.parse_args()
    if args.lower:
        cmd_lower(args)
    else:
        cmd_export_action_head(args)


if __name__ == "__main__":
    main()
