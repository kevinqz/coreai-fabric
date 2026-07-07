# /// script
# requires-python = ">=3.12"
# ///
"""GR00T-N1.7-3B action-head export lane (NVIDIA Isaac-GR00T, LeRobot v0.6.0).

GR00T N1.7 = a Cosmos-Reason2-2B VL backbone + a flow-matching DiT action head,
the SAME split discipline as EVO1 / VLA-JEPA / pi0. This lane exports the
deployable core — the per-step velocity graph:

  groot_denoise_step(x_t, timestep, vl_embeds, state_features, embodiment_id,
                     image_mask, backbone_attention_mask) -> velocity

The host owns the Cosmos-Reason2-2B backbone (+ its vl_self_attention/vlln context
prep + state encoding) and the Euler loop (num_inference_timesteps=4) +
un-normalization. Gate B is action_parity (models/groot/parity.py): fixed synthetic
VL context + fixed noise driven through the identical 4-step Euler loop.

The action head is diffusers-based (`AlternateVLDiT` = BasicTransformerBlock) +
pure-torch per-embodiment MLPs, so we rebuild it STANDALONE from the Isaac-GR00T
modules (build/_src_groot/) — dims verified against the checkpoint action_head.*
tensors — and load only `action_head.*`. Op-coverage PROVEN on coreai_torch 0.4.1
/ coremltools 9.0 (AlternateVLDiT 1.09B lowers + loads clean, no MPSGraph segfault).

NVIDIA Open Model License: derivatives may be redistributed but are NON-COMMERCIAL
(research/eval only) — the publish card/catalog must carry that + the bundled LICENSE.

Usage:
  .venv/bin/python models/groot/export.py export \
      --weights build/_groot --out build/groot-n1-7-3b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Isaac-GR00T pure-torch / diffusers modules (fetched to build/_src_groot).
GROOT_SRC = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric/build/_src_groot")

# Dims verified against the checkpoint action_head.* tensor shapes:
#   action_encoder.W1 [32,132,1536], state_encoder.layer1 [32,132,1024]/layer2[32,1024,1536],
#   action_decoder.layer2 [32,1024,132], model.attn1.to_k [1536,2048], proj_out_2 [1024,1536],
#   position_embedding [1024,1536], vlln [2048]. num_layers from final_model_config = 32.
ACTION_DIM = 132       # max_action_dim
INNER = 1536           # input_embedding_dim = num_heads(32)*head_dim(48)
HIDDEN = 1024          # DiT output_dim / hidden_size
XDIM = 2048            # backbone_embedding_dim (Cosmos-Reason2-2B hidden)
NUM_EMB = 32           # max_num_embodiments
HORIZON = 50           # action_horizon
NUM_LAYERS = 32
NH, HD = 32, 48
CTX_TOKENS = 16        # synthetic VL context length (host supplies the real one)
NUM_STEPS = 4          # num_inference_timesteps
NUM_TIMESTEP_BUCKETS = 1000


def _build_head():
    sys.path.insert(0, str(GROOT_SRC))
    import dit as gdit  # diffusers-based
    import embodiment_conditioned_mlp as emlp  # pure torch

    head = nn.Module()
    head.action_encoder = emlp.MultiEmbodimentActionEncoder(ACTION_DIM, INNER, NUM_EMB)
    head.state_encoder = emlp.CategorySpecificMLP(NUM_EMB, ACTION_DIM, HIDDEN, INNER)
    head.action_decoder = emlp.CategorySpecificMLP(NUM_EMB, HIDDEN, HIDDEN, ACTION_DIM)
    head.position_embedding = nn.Embedding(1024, INNER)
    head.vlln = nn.LayerNorm(XDIM)
    head.model = gdit.AlternateVLDiT(
        num_attention_heads=NH, attention_head_dim=HD, output_dim=HIDDEN, num_layers=NUM_LAYERS,
        dropout=0.0, norm_type="ada_norm", positional_embeddings=None,
        interleave_self_attention=True, cross_attention_dim=XDIM, attend_text_every_n_blocks=2,
    )
    return head.eval()


def _load_action_head_weights(head, weights_dir: Path) -> None:
    from safetensors import safe_open

    prefix = "action_head."
    idx = json.loads((weights_dir / "model.safetensors.index.json").read_text())["weight_map"]
    subset: dict = {}
    openers: dict = {}
    for key, shard in idx.items():
        if not key.startswith(prefix):
            continue
        if shard not in openers:
            openers[shard] = safe_open(str(weights_dir / shard), framework="pt", device="cpu")
        subset[key[len(prefix):]] = openers[shard].get_tensor(key)
    if not subset:
        raise SystemExit(f"no action-head weights with prefix {prefix!r} in {weights_dir}")
    missing, unexpected = head.load_state_dict(subset, strict=False)
    # vl_self_attention lives on the host side (context prep), so it's expectedly unused here.
    unexpected = [k for k in unexpected if not k.startswith("vl_self_attention.")]
    if unexpected:
        raise SystemExit(f"unexpected action-head keys: {unexpected[:8]}")
    real_missing = [k for k in missing if not k.startswith("vl_self_attention.")]
    if real_missing:
        raise SystemExit(f"missing action-head keys: {real_missing[:8]}")


class DenoiseStep(nn.Module):
    """One flow-matching velocity step of the GR00T action head."""

    def __init__(self, h):
        super().__init__()
        self.h = h

    def forward(self, x_t, timestep, vl_embeds, state_features, embodiment_id,
                image_mask, backbone_attention_mask):
        h = self.h
        action_features = h.action_encoder(x_t, timestep, embodiment_id)
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long,
                               device=action_features.device)
        action_features = action_features + h.position_embedding(pos_ids).unsqueeze(0)
        sa_embs = torch.cat((state_features, action_features), dim=1)
        model_output = h.model(
            hidden_states=sa_embs, encoder_hidden_states=h.vlln(vl_embeds),
            timestep=timestep, image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask)
        pred = h.action_decoder(model_output, embodiment_id)
        return pred[:, -HORIZON:]


def _example_inputs():
    b = 1
    return (
        torch.zeros(b, HORIZON, ACTION_DIM),                 # x_t
        torch.zeros(b, dtype=torch.long),                    # timestep
        torch.zeros(b, CTX_TOKENS, XDIM),                    # vl_embeds
        torch.zeros(b, 1, INNER),                            # state_features
        torch.zeros(b, dtype=torch.long),                    # embodiment_id
        torch.ones(b, CTX_TOKENS, dtype=torch.bool),         # image_mask
        torch.ones(b, CTX_TOKENS, dtype=torch.bool),         # backbone_attention_mask
    )


def cmd_export(args) -> None:
    from coreai_torch import TorchConverter, get_decomp_table

    head = _build_head()
    _load_action_head_weights(head, args.weights)
    wrapper = DenoiseStep(head).eval()

    with torch.no_grad():
        ep = torch.export.export(wrapper, args=_example_inputs(), strict=False)
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(
        ep,
        input_names=["x_t", "timestep", "vl_embeds", "state_features", "embodiment_id",
                     "image_mask", "backbone_attention_mask"],
        output_names=["velocity"], entrypoint_name="groot_denoise_step")
    prog = conv.to_coreai()
    prog.optimize()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    aimodel = out / f"{out.name}.aimodel"
    prog.save_asset(aimodel)
    contract = {
        "entrypoint": "groot_denoise_step",
        "action_dim": ACTION_DIM, "inner": INNER, "hidden": HIDDEN, "backbone_embedding_dim": XDIM,
        "num_embodiments": NUM_EMB, "action_horizon": HORIZON, "num_inference_timesteps": NUM_STEPS,
        "ctx_tokens": CTX_TOKENS,
        "host_components": ["cosmos_reason2_2b_backbone", "vl_self_attention", "state_encoder",
                            "euler_loop", "un_normalize"],
    }
    (out / "groot-action-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"ok: lowered GR00T action head -> {aimodel}")
    print(f"ok: wrote {out / 'groot-action-contract.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="GR00T-N1.7 action-head export")
    ap.add_argument("phase", nargs="?", default="export", choices=["export"])
    ap.add_argument("--weights", type=Path, required=True, help="GR00T checkpoint dir (sharded safetensors + index)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cmd_export(args)


if __name__ == "__main__":
    main()
