"""Miril-Drone-2B-1 — gemma4_vision encoder: lower the transferred ExportedProgram +
graph_output_cosine parity.

Gemma-4 needs transformers 5.x (conflicts with the toolchain's <5.0 pin) and ships no
modeling code, so the REAL Gemma4VisionModel graph is captured in a scratch venv
(transformers 5.13, torch 2.9 == toolchain torch) via torch.export.save -> vision.pt2,
alongside the fp32 reference (refs.npy) and the fixed inputs (inputs.pt). See
models/miril/ref_export.py. This driver loads that ExportedProgram in the toolchain
venv and lowers it with coreai_torch — bit-exact by construction (the real impl's
clipped-linears / RoPE / sandwich-norms are all in the captured aten graph).

Deployable graph = image patches (pixel_values [1,2520,768] + pixel_position_ids
[1,2520,2]) -> pooled vision soft-tokens [256,768]. Host owns the image processor
(patchify) + the Gemma-4 LM. fp32. Gate B = graph_output_cosine of the .aimodel vs the
scratch-venv Gemma4VisionModel reference over n_obs=8 seeded images."""
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/Users/kevinsaltarelli/Dev/Github/coreai-fabric")
REF = Path("/private/tmp/claude-501/-Users-kevinsaltarelli-Dev-Github-coreai-fabric--claude-worktrees-beautiful-dhawan-6be758/a1dd52aa-599c-41ba-8f12-bac547994f54/scratchpad/miril_out")
OUT = ROOT / "build/miril-drone-2b-1"
N_OBS = 8


def main():
    from coreai_torch import TorchConverter, get_decomp_table

    ep = torch.export.load(str(REF / "vision.pt2"))
    print("loaded ExportedProgram from scratch venv", flush=True)
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(ep, input_names=["pixel_values"],
                              output_names=["vision_tokens"], entrypoint_name="vision_encode")
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
        mm = await AIModel.load(str(aim)); fn = mm.load_function("vision_encode")
        print("LOADS OK", flush=True)
        dt_pv = np.dtype(str(fn.desc.input_descriptor("pixel_values").dtype))
        cos = []
        for i in range(N_OBS):
            pv = inputs["pixel_values"][i].numpy().astype(dt_pv)
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
               "reference": ("Real transformers-5.13 Gemma4VisionModel (fp32) last_hidden_state vs the "
                             "coreai .aimodel over 8 seeded 896x896 images. ExportedProgram transferred "
                             "from a scratch venv; the toolchain venv only lowers it. Clipped-linears / "
                             "RoPE / sandwich-norms captured from the real impl.")}
        (OUT / "graph-output-parity-measured.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({k: res[k] for k in ("value", "min_cosine", "median_cosine", "n_obs")}, indent=2))
    asyncio.run(run())
    print("DONE", flush=True)


main()
