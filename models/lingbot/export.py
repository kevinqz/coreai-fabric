# /// script
# requires-python = ">=3.12"
# ///
"""LingBot-Vision export lane (robbyant/lingbot-vision ViT backbones).

A NON-autoregressive vision encoder: `image [1,3,S,S] -> normalized per-patch
tokens [1, (S/16)^2, embed_dim]`. One static-size `.aimodel`, single entrypoint
`main`. The host owns image preprocessing (resize to S, ImageNet mean/std; see
the upstream `lingbot_vision.preprocess.load_image`). Gate B is
graph_output_cosine (models/lingbot/parity.py) — the vision analog of the LLM
logit-parity: identical seeded images through the torch reference and the
lowered asset, compared by cosine on the patch tokens.

Op-coverage proven on coreai_torch 0.4.1 / coremltools 9.0 (ViT-S; SDPA attention
lowers). The upstream model class + weights come from the robbyant/lingbot-vision
repo checkout (only `model.pt` ships on HF; the repo carries the ViT code), so
this is a per-model script lane like depth-anything.

Usage (single env; the lingbot-vision repo on sys.path via --lingbot-src):
  .venv/bin/python models/lingbot/export.py export \
      --variant small --out build/lingbot-vision-vit-small \
      --lingbot-src build/_src_lingbot --size 512
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

DEFAULT_SIZE = 512
PATCH = 16


def _load_backbone(variant: str, lingbot_src: Path):
    """Load a frozen LingBot-Vision ViT backbone via the upstream repo loader."""
    sys.path.insert(0, str(lingbot_src))
    from lingbot_vision import load_pretrained_backbone

    model, embed_dim = load_pretrained_backbone(variant=variant, device="cpu", dtype="fp32")
    return model.eval(), int(embed_dim)


class PatchTokens(torch.nn.Module):
    """Expose the normalized per-patch tokens (drop the CLS/storage tokens + dict)."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, image):
        out = self.backbone.forward_features(image)
        return out["x_norm_patchtokens"] if isinstance(out, dict) else out


def cmd_export(args) -> None:
    from coreai_torch import TorchConverter, get_decomp_table

    backbone, embed_dim = _load_backbone(args.variant, args.lingbot_src)
    wrapper = PatchTokens(backbone).eval()
    size = int(args.size) // PATCH * PATCH  # snap to a multiple of the patch size
    image = torch.zeros(1, 3, size, size, dtype=torch.float32)

    with torch.no_grad():
        ref = wrapper(image)
    with torch.no_grad():
        ep = torch.export.export(wrapper, args=(image,), strict=False)
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(
        ep, input_names=["image"], output_names=["patch_tokens"], entrypoint_name="main")
    prog = conv.to_coreai()
    prog.optimize()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    aimodel = out / f"{out.name}.aimodel"
    prog.save_asset(aimodel)
    contract = {
        "entrypoint": "main",
        "image_size": size,
        "patch_size": PATCH,
        "num_patch_tokens": int(ref.shape[-2]),
        "embed_dim": embed_dim,
        "variant": args.variant,
        "host_components": ["image_resize", "imagenet_normalize"],
    }
    (out / "lingbot-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"ok: lowered LingBot ViT-{args.variant} (embed_dim={embed_dim}, "
          f"{int(ref.shape[-2])} patch tokens @ {size}px) -> {aimodel}")
    print(f"ok: wrote {out / 'lingbot-contract.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="LingBot-Vision ViT export (image -> patch tokens)")
    ap.add_argument("phase", nargs="?", default="export", choices=["export"])
    ap.add_argument("--variant", required=True, choices=["small", "base", "large", "giant"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lingbot-src", type=Path, default=Path("build/_src_lingbot"),
                    help="checkout of github.com/robbyant/lingbot-vision (carries the ViT code)")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE)
    args = ap.parse_args()
    cmd_export(args)


if __name__ == "__main__":
    main()
