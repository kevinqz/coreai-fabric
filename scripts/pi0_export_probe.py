"""pi0 op-coverage probe — answers the two UNCERTAINs before any full export.

Cheap, disk-light: exports + lowers ONE subgraph at a time and reports which ops
(if any) fail to lower, plus the resolved prefix_len (the shape gate).

ORDER MATTERS — run `denoise` FIRST, then `encode`:
  - denoise : smaller graph (Gemma-300m expert, NO vision tower). Isolates the
              flow-matching / attention-topology ops (RoPE, RMSNorm, GQA-SDPA,
              time embed, the block-causal mask). If the block mask lowers here
              (dense additive tensor via coreai_torch replace_sdpa), the whole
              LLM+action path is green. It also PRINTS the real prefix_len ->
              resolves GATE 2a (num_img_embs 256 -> 816, or pooled 1 -> 51).
  - encode  : carries the ONE true unknown (SigLIP/PaliGemma on coremltools 9.0).
              Probing it second means a denoise failure is diagnosed without the
              vision tower's noise. This is GATE 2b (the biggest risk).

This is a TWO-VENV probe like the export: torch.export in venv-A, lower in venv-B.
For a single-venv smoke test (op-trace only, no lower) run with --export-only in venv-A.

Usage:
  .venv-lerobot/bin/python scripts/pi0_export_probe.py denoise --export-only   # venv-A: trace + shape
  .venv/bin/python         scripts/pi0_export_probe.py denoise --lower <tmp>   # venv-B: lower the .pt2
  ... then repeat for `encode`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def probe_export(which: str, tmp: Path) -> dict:
    """venv-A: build the wrapper, torch.export it, print the resolved shapes, save .pt2."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "pi0"))
    import export as pi0_export  # models/pi0/export.py

    enc, den, d = pi0_export._build_wrappers()
    tmp.mkdir(parents=True, exist_ok=True)
    if which == "denoise":
        ep = torch.export.export(den, args=(d["state"], d["ppad"], d["xt"], d["t"], *d["cache"]),
                                 strict=False)
        # resolve prefix_len from a real encode forward (GATE 2a)
        with torch.no_grad():
            enc_out = enc(d["img"], d["img"], d["img"], d["imask"], d["imask"], d["imask"],
                          d["tok"], d["lmask"])
        prefix_len = int(enc_out[0].shape[1])
    else:
        ep = torch.export.export(
            enc, args=(d["img"], d["img"], d["img"], d["imask"], d["imask"], d["imask"],
                       d["tok"], d["lmask"]), strict=False)
        prefix_len = None
    torch.export.save(ep, str(tmp / f"{which}.pt2"))
    return {"subgraph": which, "exported": True, "prefix_len": prefix_len,
            "note": "prefix_len resolves GATE 2a (816=256/img patches, 51=pooled)"}


def probe_lower(which: str, tmp: Path) -> dict:
    """venv-B: load the .pt2 and try to lower it; report the first unsupported op."""
    import torch
    from coreai_torch import TorchConverter, get_decomp_table
    ep = torch.export.load(str(tmp / f"{which}.pt2"))
    try:
        ep = ep.run_decompositions(get_decomp_table())
        TorchConverter().add_exported_program(ep).to_coreai()
        return {"subgraph": which, "lowered": True, "first_unsupported_op": None}
    except Exception as exc:  # noqa: BLE001 — the point is to REPORT the failure, not raise
        return {"subgraph": which, "lowered": False,
                "first_unsupported_op": f"{type(exc).__name__}: {str(exc)[:300]}"}


def main():
    ap = argparse.ArgumentParser(description="pi0 op-coverage probe (denoise first, then encode)")
    ap.add_argument("which", choices=["denoise", "encode"])
    ap.add_argument("--export-only", action="store_true", help="venv-A: trace + shape, no lower")
    ap.add_argument("--lower", type=Path, metavar="TMPDIR", help="venv-B: lower the .pt2 in TMPDIR")
    args = ap.parse_args()
    tmp = args.lower or Path("/tmp/pi0_probe")
    result = probe_lower(args.which, tmp) if args.lower else probe_export(args.which, tmp)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
