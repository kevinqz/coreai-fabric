"""Concat the 8 range chunks -> reconstruct action_expert.safetensors (strip prefix)."""
import json, os, glob, numpy as np
os.chdir("/Users/kevinsaltarelli/Dev/Github/coreai-fabric")
plan = json.load(open("build/_molmoact2/ae_plan.json"))
ts = plan["tensors"]; base = plan["base"]
offs = [v["data_offsets"] for v in ts.values()]
lo = min(a for a, b in offs)
chunks = sorted(glob.glob("build/_molmoact2/chunk_*.bin"), key=lambda p: int(p.split("_")[-1].split(".")[0]))
assert len(chunks) == 8, f"expected 8 chunks, got {len(chunks)}"
data = b"".join(open(c, "rb").read() for c in chunks)
span = max(b for a, b in offs) - lo
assert len(data) == span, f"blob {len(data)} != span {span}"
print(f"concatenated {len(data)/1e9:.2f}GB")
from safetensors.numpy import save_file
sd = {}; PRE = "model.model.action_expert."
for k, v in ts.items():
    a, b = v["data_offsets"]; sh = v["shape"]
    seg = data[a - lo:b - lo]
    arr = np.frombuffer(seg, dtype="<f4")
    arr = arr.reshape(sh).copy() if sh else arr.copy()
    sd[k[len(PRE):]] = arr
save_file(sd, "build/_molmoact2/action_expert.safetensors")
print(f"reconstructed {len(sd)} tensors -> action_expert.safetensors ({os.path.getsize('build/_molmoact2/action_expert.safetensors')/1e9:.2f}GB)")
for c in chunks:
    os.remove(c)
print("freed chunks. DONE")
