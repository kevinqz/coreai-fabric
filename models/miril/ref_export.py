"""Scratch-venv (transformers 5.13) — load the REAL Gemma4 vision tower, produce the
fp32 reference, and torch.export it for transfer to the toolchain venv."""
import glob, struct, json
from pathlib import Path
import numpy as np
import torch
from transformers import Gemma4VisionModel, Gemma4ImageProcessor, AutoConfig

SNAP = Path.home() / ".cache/huggingface/hub/models--MirilAI--Miril-Drone-2B-1/snapshots/40bde4fe61edc451440f62dc6653ddb06999543d"
OUT = Path("/private/tmp/claude-501/-Users-kevinsaltarelli-Dev-Github-coreai-fabric--claude-worktrees-beautiful-dhawan-6be758/a1dd52aa-599c-41ba-8f12-bac547994f54/scratchpad/miril_out")
OUT.mkdir(exist_ok=True)
N_OBS, SEED = 8, 0

cfg = AutoConfig.from_pretrained(str(SNAP)).vision_config
model = Gemma4VisionModel(cfg).eval()

# load vision_tower weights from the safetensors shards (strip the model.vision_tower. prefix)
from safetensors.torch import load_file
sd = {}
for shard in glob.glob(str(SNAP / "*.safetensors")):
    raw = load_file(shard)
    for k, v in raw.items():
        if k.startswith("model.vision_tower."):
            sd[k[len("model.vision_tower."):]] = v.float()
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"loaded {len(sd)} vision tensors; missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
if missing:
    print("  MISSING:", missing[:6], flush=True)
if unexpected:
    print("  UNEXPECTED:", unexpected[:6], flush=True)

# build inputs with the real image processor on a fixed-size seeded image
proc = Gemma4ImageProcessor.from_pretrained(str(SNAP))
def make_inputs(i):
    g = np.random.RandomState(SEED + i)
    img = (g.rand(896, 896, 3) * 255).astype("uint8")
    from PIL import Image
    enc = proc(images=Image.fromarray(img), return_tensors="pt")
    return enc

enc0 = make_inputs(0)
print("processor output keys:", list(enc0.keys()), flush=True)
for k, v in enc0.items():
    if hasattr(v, "shape"):
        print(f"  {k}: {tuple(v.shape)} {v.dtype}", flush=True)

# the processor emits `image_position_ids`; the model forward wants `pixel_position_ids`
pid_key = "image_position_ids"
print("mapping", pid_key, "-> pixel_position_ids", flush=True)

POOL = model.config.pooling_kernel_size

def pooled_forward(m, pixel_values, pixel_position_ids):
    """The VisionModel forward UP TO the pooler (static (1, N_pooled, 768)) — skips the
    final data-dependent padding-drop (constant for a fixed image size; host-owned)."""
    output_length = pixel_values.shape[-2] // (POOL * POOL)
    padding_positions = (pixel_position_ids == -1).all(dim=-1)
    inputs_embeds = m.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
    enc = m.encoder(inputs_embeds=inputs_embeds, attention_mask=~padding_positions,
                    pixel_position_ids=pixel_position_ids)
    pooled, _ = m.pooler(hidden_states=enc.last_hidden_state, pixel_position_ids=pixel_position_ids,
                         padding_positions=padding_positions, output_length=output_length)
    return pooled  # (1, output_length, 768), static

def call(enc):
    with torch.no_grad():
        out = pooled_forward(model, enc["pixel_values"].float(), enc[pid_key])
    return out.float().numpy()

refs = []
encs = [make_inputs(i) for i in range(N_OBS)]
for e in encs:
    refs.append(call(e))
print("ref OK, last_hidden_state shape:", refs[0].shape, flush=True)
np.save(OUT / "refs.npy", np.stack(refs))
# save the input tensors (fixed shape) for reuse in the toolchain venv
torch.save({"pixel_values": [e["pixel_values"].float() for e in encs]}, OUT / "inputs.pt")

# torch.export for transfer. The image size is FIXED (896x896 -> 2520 patches), so
# pixel_position_ids is a CONSTANT grid — bake it as a buffer so the position-embedding
# gather constant-folds (coreai_torch can't lower the rank-4 advanced-index otherwise).
# Deployable interface becomes pixel_values-only; the host uses the same fixed grid.
assert all(torch.equal(encs[0][pid_key], e[pid_key]) for e in encs), "position grid must be constant across obs"

class Wrap(torch.nn.Module):
    def __init__(s, m, pid):
        super().__init__(); s.m = m; s.register_buffer("pid", pid)
    def forward(s, pixel_values):
        return pooled_forward(s.m, pixel_values, s.pid)

w = Wrap(model, encs[0][pid_key]).eval()
args = (encs[0]["pixel_values"].float(),)
try:
    ep = torch.export.export(w, args=args, strict=False)
    torch.export.save(ep, str(OUT / "vision.pt2"))
    print("EXPORT+SAVE OK ->", OUT / "vision.pt2", flush=True)
    print("input_names: pixel_values (position grid baked as constant)", flush=True)
except Exception as e:
    import traceback; print("EXPORT FAIL:", type(e).__name__, str(e)[:400]); traceback.print_exc()
print("DONE", flush=True)
