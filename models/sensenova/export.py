"""SenseNova-Vision-7B-MoT — lower the transferred SigLIP ExportedProgram +
graph_output_cosine parity (toolchain venv: coreai_torch 0.4.1 / coremltools 9).

⚠️ DRAFT / UNVERIFIED — NOT YET RUN. Mirrors models/miril/export.py. Run
models/sensenova/ref_export.py FIRST (scratch venv) to produce build/_sensenova/
{vision.pt2, refs.npy, inputs.pt}, then run this in the toolchain venv on
macOS 27. Iterate against Gate B (graph_output_cosine) — no number is fabricated.

Deployable graph = pixel_values [1,3,980,980] -> per-patch vision tokens
[1,4900,1152] (entrypoint `main`). Host owns SigLIP preprocessing + the MoT
Qwen2 decoder + FLUX VAE. fp32. INDEX-ONLY (CC-BY-NC-4.0): local asset + a
measured fidelity number are the deliverable; weights are NOT republished.

Run:
  python models/sensenova/export.py
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric")
REF = ROOT / "build/_sensenova"          # produced by ref_export.py
OUT = ROOT / "build/sensenova-vision-7b-mot"
N_OBS = 8


def main():
    from coreai_torch import TorchConverter, get_decomp_table

    ep = torch.export.load(str(REF / "vision.pt2"))
    print("loaded SigLIP ExportedProgram from scratch venv", flush=True)
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(ep, input_names=["pixel_values"],
                              output_names=["vision_tokens"], entrypoint_name="main")
    prog = conv.to_coreai(); prog.optimize()
    OUT.mkdir(parents=True, exist_ok=True)
    aim = OUT / f"{OUT.name}.aimodel"; prog.save_asset(aim)
    sz = sum(f.stat().st_size for f in aim.rglob("*") if f.is_file())
    print(f"ok: saved fp32 {aim} (~{sz/1e6:.1f} MB)", flush=True)
    import gc; del prog, conv, ep; gc.collect()

    inputs = torch.load(REF / "inputs.pt")
    refs = np.load(REF / "refs.npy")

    import asyncio
    from coreai.runtime import AIModel, NDArray
    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _environment

    async def run():
        mm = await AIModel.load(str(aim)); fn = mm.load_function("main")
        print("LOADS OK", flush=True)
        dt = np.dtype(str(fn.desc.input_descriptor("pixel_values").dtype))
        cos = []
        for i in range(N_OBS):
            pv = inputs["pixel_values"][i].numpy().astype(dt)
            out = await fn(inputs={"pixel_values": NDArray(pv)})
            a = out["vision_tokens"].numpy().astype(np.float64).reshape(-1)
            b = refs[i].astype(np.float64).reshape(-1)
            cos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        cc = np.asarray(cos)
        res = {"metric": "graph_output_cosine", "value": float(cc.min()), "status": "measured",
               "min_cosine": float(cc.min()), "median_cosine": float(np.median(cc)),
               "mean_cosine": float(cc.mean()), "per_obs_cosine": [float(x) for x in cos],
               "n_obs": N_OBS, "reference_dtype": "float32", "quantization": "none",
               "asset_bytes": int(sz), "output_shape": list(refs[0].shape),
               "runner": f"coreai-fabric-parity-runner/{__version__}", "environment": _environment(),
               "reference": ("Real transformers SiglipVisionModel (fp32) last_hidden_state vs the "
                             "coreai .aimodel over 8 seeded 980x980 images. ExportedProgram transferred "
                             "from a scratch venv; the toolchain venv only lowers it.")}
        (OUT / "graph-output-parity-measured.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({k: res[k] for k in ("value", "min_cosine", "median_cosine", "n_obs")}, indent=2))
    asyncio.run(run())
    print("DONE", flush=True)


main()
