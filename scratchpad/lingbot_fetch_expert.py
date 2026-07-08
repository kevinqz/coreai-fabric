"""Fetch the qwen_expert (+ action heads) weights from robbyant/lingbot-vla-v2-6b
via per-shard envelope range requests (6 shards, ~7.3GB), reconstruct a clean
qwen_expert.safetensors. Skips the 17.75GB Qwen3-VL backbone (host-side)."""
import urllib.request, json, struct, os, time, concurrent.futures as cf
from pathlib import Path
import torch
os.chdir("/Users/kevinsaltarelli/Dev/Github/coreai-fabric")
BASE = "https://huggingface.co/robbyant/lingbot-vla-v2-6b/resolve/main/"
OUT = Path("build/_lingbotvla"); OUT.mkdir(parents=True, exist_ok=True)
TORCH_DT = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
            "I64": torch.int64, "I32": torch.int32, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


def need(k):
    return ("qwen_expert" in k) or k.startswith(("model.action_in_proj", "model.action_out_proj", "model.action_time_mlp", "model.state_proj"))


def rng(url, a, b, retries=8):
    for i in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"}), timeout=180).read()
        except Exception as e:
            print(f"  retry {i} {url[-25:]} @{a}: {e}", flush=True); time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed {url} {a}-{b}")


def fetch_span(url, a, b, chunk=48 * 1024 * 1024):
    buf = bytearray()
    pos = a
    while pos <= b:
        end = min(pos + chunk - 1, b)
        buf += rng(url, pos, end); pos = end + 1
    return bytes(buf)


# build plan
idx = json.load(urllib.request.urlopen(BASE + "model.safetensors.index.json"))
wm = idx["weight_map"]
shards = sorted(set(wm.values()))
plan = {}
for sh in shards:
    url = BASE + sh
    n = struct.unpack("<Q", rng(url, 0, 7))[0]
    hdr = json.loads(rng(url, 8, 8 + n - 1)); base = 8 + n
    keys = [k for k in hdr if k != "__metadata__" and need(k)]
    if not keys:
        continue
    los = [hdr[k]["data_offsets"][0] for k in keys]; his = [hdr[k]["data_offsets"][1] for k in keys]
    lo, hi = min(los), max(his)
    plan[sh] = {"url": url, "base": base, "env_lo": lo, "env_hi": hi,
                "tensors": {k: {"off": hdr[k]["data_offsets"], "shape": hdr[k]["shape"], "dtype": hdr[k]["dtype"]} for k in keys}}
    print(f"plan {sh}: {len(keys)} tensors, env {(hi-lo)/1e9:.2f}GB", flush=True)

# fetch envelopes in parallel
t = time.time()
def dl(sh):
    p = plan[sh]; a = p["base"] + p["env_lo"]; b = p["base"] + p["env_hi"] - 1
    data = fetch_span(p["url"], a, b)
    print(f"  fetched {sh} {len(data)/1e9:.2f}GB ({time.time()-t:.0f}s)", flush=True)
    return sh, data
blobs = {}
with cf.ThreadPoolExecutor(max_workers=6) as ex:
    for sh, data in ex.map(dl, list(plan.keys())):
        blobs[sh] = data
print(f"all envelopes fetched ({time.time()-t:.0f}s)", flush=True)

# reconstruct state_dict (original keys, strip leading 'model.')
sd = {}
for sh, p in plan.items():
    data = blobs[sh]; lo = p["env_lo"]
    for k, meta in p["tensors"].items():
        a, b = meta["off"]; seg = data[a - lo:b - lo]
        dt = TORCH_DT[meta["dtype"]]
        tt = torch.frombuffer(bytearray(seg), dtype=dt)
        tt = tt.reshape(meta["shape"]) if meta["shape"] else tt.reshape(())
        sd[k[len("model."):]] = tt.clone()
from safetensors.torch import save_file
save_file(sd, str(OUT / "qwen_expert.safetensors"))
print(f"reconstructed {len(sd)} tensors -> qwen_expert.safetensors ({os.path.getsize(OUT/'qwen_expert.safetensors')/1e9:.2f}GB)", flush=True)
json.dump({sh: {k: v for k, v in p.items() if k != "url"} for sh, p in plan.items()}, open(OUT / "expert_plan.json", "w"), default=str)
print("DONE", flush=True)
