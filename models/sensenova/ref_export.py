"""SenseNova-Vision-7B-MoT — SigLIP vision-encoder reference capture (scratch venv).

⚠️ DRAFT / UNVERIFIED — NOT YET RUN. This scaffolds the two-venv encoder lane
(mirrors models/miril/ref_export.py). It MUST be run + iterated on-device
(macOS 27) against the real sensenova/SenseNova-Vision-7B-MoT checkpoint before
it produces a valid asset; the two crux unknowns it self-diagnoses at runtime
are flagged TODO below. INDEX-ONLY (CC-BY-NC-4.0): the deliverable is this
reproducible recipe + a measured Gate-B number, never republished weights.

Deployable core = the SigLIP-so400m vision encoder (understanding path):
  image [1,3,980,980] -> per-patch tokens [1, (980/14)^2=4900, 1152].
The host owns the SigLIP image preprocessing + the MoT Qwen2 decoder + the FLUX
VAE generation path. SenseNova's encoder is a STANDARD transformers
`SiglipVisionModel` (config.vision_config.model_type == 'siglip_vision_model'),
so no custom modeling code is needed — only the weight slice + the right config.

Why two venvs (same rationale as miril / the LLM logit lanes): the reference
stack (transformers for SenseNova/BAGEL) and coreai_torch's pinned torch/
transformers can conflict, so we capture the fp32 reference + a torch.export
ExportedProgram HERE and lower it in the toolchain venv (models/sensenova/
export.py) — bit-exact by construction.

Run (scratch venv with transformers + safetensors + pillow):
  python models/sensenova/ref_export.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import SiglipVisionConfig, SiglipVisionModel

# The 29.2GB ema.safetensors is CC-BY-NC — do NOT commit/redistribute. Point at
# the local HF snapshot; range-fetch only the vision slice if disk-constrained.
SNAP = Path.home() / ".cache/huggingface/hub/models--sensenova--SenseNova-Vision-7B-MoT"
OUT = Path("build/_sensenova")   # refs + inputs + vision.pt2 for the toolchain venv
OUT.mkdir(parents=True, exist_ok=True)
N_OBS, SEED, IMG = 8, 0, 980
PATCH = 14

# SigLIP-so400m config from the recipe analysis (arXiv:2607.06560): hidden 1152,
# 27 layers, patch-14, 980px. TODO(on-device): confirm num_attention_heads /
# intermediate_size against config.vision_config (so400m is 16 heads / 4304 FFN).
cfg = SiglipVisionConfig(
    hidden_size=1152, num_hidden_layers=27, num_attention_heads=16,
    intermediate_size=4304, num_channels=3, image_size=IMG, patch_size=PATCH,
)
model = SiglipVisionModel(cfg).eval()

# TODO(on-device): the vision-tower weight-key PREFIX inside ema.safetensors is
# unknown without inspecting the checkpoint (BAGEL nests SigLIP under e.g.
# `vit_model.`, `vision_model.`, or `understanding.vision.`). The loop tries the
# common ones and self-diagnoses via missing/unexpected — pick the prefix that
# yields 0 missing. This is the #1 thing to resolve on the first real run.
CANDIDATE_PREFIXES = ["vit_model.", "vision_model.", "model.vision_model.",
                      "understanding.vision_model.", "siglip.vision_model."]
shards = glob.glob(str(SNAP / "snapshots" / "*" / "*.safetensors"))
assert shards, f"no safetensors under {SNAP} — download the checkpoint first"

best = None
for prefix in CANDIDATE_PREFIXES:
    sd = {}
    for shard in shards:
        for k, v in load_file(shard).items():
            if k.startswith(prefix):
                sd[k[len(prefix):]] = v.float()
    if not sd:
        continue
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"prefix {prefix!r}: {len(sd)} tensors, missing {len(missing)}, unexpected {len(unexpected)}",
          flush=True)
    if best is None or len(missing) < best[1]:
        best = (prefix, len(missing), len(unexpected))
assert best is not None, "no candidate prefix matched — inspect the checkpoint keys"
print(f"best prefix: {best[0]!r} (missing {best[1]})", flush=True)
# Re-load the best prefix definitively.
prefix = best[0]
sd = {}
for shard in shards:
    for k, v in load_file(shard).items():
        if k.startswith(prefix):
            sd[k[len(prefix):]] = v.float()
model.load_state_dict(sd, strict=False)

# Seeded fixed-size images -> pixel_values [1,3,980,980] (host preprocessing is
# SigLIP standard: resize 980, rescale 1/255, normalize mean/std 0.5). Kept
# self-contained (no image processor dep) so the reference is reproducible.
def make_pixel_values(i: int) -> torch.Tensor:
    g = np.random.RandomState(SEED + i)
    img = g.rand(3, IMG, IMG).astype("float32")     # already [0,1)
    px = (img - 0.5) / 0.5                            # SigLIP normalize
    return torch.from_numpy(px).unsqueeze(0)          # [1,3,980,980]

encs = [make_pixel_values(i) for i in range(N_OBS)]

def call(pv: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = model(pixel_values=pv).last_hidden_state   # [1, 4900, 1152]
    return out.float().numpy()

refs = [call(e) for e in encs]
print("ref OK, last_hidden_state shape:", refs[0].shape, flush=True)
np.save(OUT / "refs.npy", np.stack(refs))
torch.save({"pixel_values": encs}, OUT / "inputs.pt")

# torch.export for transfer. Static image size -> a single fixed graph.
# TODO(on-device): if SigLIP SDPA hits the macOS 27 FoldMultiplyIntoSDPAScale
# segfault, inject a data-independent all-true bool keep-mask (see the recipe
# notes) before lowering in export.py.
class Wrap(torch.nn.Module):
    def __init__(s, m):
        super().__init__(); s.m = m
    def forward(s, pixel_values):
        return s.m(pixel_values=pixel_values).last_hidden_state

w = Wrap(model).eval()
try:
    ep = torch.export.export(w, args=(encs[0],), strict=False)
    torch.export.save(ep, str(OUT / "vision.pt2"))
    print("EXPORT+SAVE OK ->", OUT / "vision.pt2", flush=True)
except Exception as e:  # noqa: BLE001
    import traceback
    print("EXPORT FAIL:", type(e).__name__, str(e)[:400]); traceback.print_exc()
(OUT / "capture-meta.json").write_text(json.dumps(
    {"prefix": prefix, "image_size": IMG, "patch": PATCH, "n_obs": N_OBS,
     "hidden": cfg.hidden_size, "layers": cfg.num_hidden_layers}, indent=2) + "\n")
print("DONE", flush=True)
