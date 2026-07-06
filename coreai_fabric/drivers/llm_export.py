"""`coreai-fabric-llm-export` — fabric's own LLM converter executable.

Validated end-to-end on real hardware (Apple Silicon, macOS 26,
coreai-torch 0.4.1 + coreai-core 1.0.0b2 from PyPI, transformers 4.57.3):
torch.export -> TorchConverter -> to_coreai() -> optimize() -> save_asset()
produces a loadable `.aimodel` whose outputs match the upstream PyTorch
model (cosine 1.0 on a smoke model; see docs/validation-log.md for the
qwen3-0.6b run).

Scope — honest limits:
- STATIC single-graph export: one `main` function, fixed (1, seq_len)
  `input_ids` -> full-sequence `logits`. No KV-cache states, no
  prefill/decode split. Correct for parity verification and batch scoring;
  NOT the production chat-asset layout that Apple's `coreai.llm.export`
  (apple/coreai-models checkout, not on PyPI) produces.
- `--compression none` only. Quantized exports need Apple's pipeline
  (coreai-opt), which fabric does not reimplement.
- Flag layout mirrors the verified `coreai.llm.export` interface (positional
  model id, --output-dir/--output-name/--compute-precision/--compression/
  --platform/--overwrite) plus fabric extensions Apple's CLI lacks:
  --revision (upstream pinning) and --seq-len.

Output layout (verified identical to Apple's pipeline convention):
  <output-dir>/<output-name>/<output-name>.aimodel/   (main.mlirb, main.hash, metadata.json)
  <output-dir>/<output-name>/tokenizer/               (upstream tokenizer)
  <output-dir>/<output-name>/metadata.json            (bundle-level manifest)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

__version__ = "0.1.0"

PRECISIONS = ("float16", "bfloat16", "float32")

MISSING_STACK_HINT = """\
coreai-fabric-llm-export requires the Apple conversion stack (macOS on Apple
Silicon). Install fabric's convert extra:

    pip install "coreai-fabric[convert]"

which pins the stack verified on real hardware: coreai-torch==0.4.1
(pulls coreai-core==1.0.0b2) plus transformers/torch. All of it is on PyPI;
no apple/coreai-models checkout is needed for this driver.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coreai-fabric-llm-export",
        description="Convert a Hugging Face causal LM to a Core AI .aimodel "
        "(static graph) via the coreai-torch library.",
    )
    parser.add_argument("model", help="Hugging Face model id (owner/name)")
    parser.add_argument("--output-dir", required=True, help="Directory to place the bundle dir in")
    parser.add_argument("--output-name", required=True, help="Bundle dir and .aimodel base name")
    parser.add_argument("--compute-precision", choices=PRECISIONS, default="float16")
    parser.add_argument(
        "--compression",
        default="none",
        help="Only 'none' is supported by this driver (quantized exports need "
        "Apple's coreai.llm.export from an apple/coreai-models checkout)",
    )
    parser.add_argument(
        "--platform",
        choices=["macOS"],
        default="macOS",
        help="Kept for flag-compatibility with coreai.llm.export; this driver "
        "only produces the macOS-style dynamic asset",
    )
    parser.add_argument("--revision", default=None, help="Upstream HF revision (commit sha) to pin")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=96,
        help="Static sequence length of the exported graph (default: 96)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--version",
        action="version",
        version=f"coreai-fabric-llm-export {__version__}",
    )
    return parser


def _stack_versions() -> dict:
    """Exact versions of the conversion stack in this environment. Never
    fabricated: only reports importable distributions."""
    from importlib import metadata

    versions = {}
    for dist in ("coreai-torch", "coreai-core", "torch", "transformers"):
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    return versions


def export(args) -> Path:
    try:
        import torch
        from coreai_torch import TorchConverter, get_decomp_table
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(f"error: missing conversion stack: {exc}", file=sys.stderr)
        print(MISSING_STACK_HINT, file=sys.stderr)
        raise SystemExit(1) from exc

    if args.compression != "none":
        raise SystemExit(
            f"error: --compression {args.compression!r} is not supported by "
            "coreai-fabric-llm-export (only 'none'). Use Apple's coreai.llm.export "
            "from an apple/coreai-models checkout for quantized exports."
        )

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        args.compute_precision
    ]

    bundle_dir = Path(args.output_dir) / args.output_name
    aimodel_path = bundle_dir / f"{args.output_name}.aimodel"
    if aimodel_path.exists() and not args.overwrite:
        raise SystemExit(f"error: {aimodel_path} already exists (pass --overwrite)")

    print(f"[fabric] loading {args.model}" + (f" @ {args.revision}" if args.revision else ""))
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=dtype
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)

    class LogitsWrapper(torch.nn.Module):
        """Static forward: (1, seq_len) input_ids -> (1, seq_len, vocab) logits."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids):
            return self.inner(input_ids=input_ids, use_cache=False).logits

    torch.manual_seed(0)
    example_ids = torch.randint(
        low=0, high=int(model.config.vocab_size), size=(1, args.seq_len), dtype=torch.long
    )

    print(f"[fabric] torch.export (static seq_len={args.seq_len}) ...")
    wrapper = LogitsWrapper(model)
    with torch.no_grad():
        exported = torch.export.export(wrapper, args=(example_ids,))
    exported = exported.run_decompositions(get_decomp_table())

    print("[fabric] converting to Core AI IR ...")
    converter = TorchConverter().add_exported_program(
        exported,
        input_names=["input_ids"],
        output_names=["logits"],
    )
    program = converter.to_coreai()
    program.optimize()

    print(f"[fabric] saving asset to {aimodel_path} ...")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    program.save_asset(aimodel_path)

    tokenizer.save_pretrained(str(bundle_dir / "tokenizer"))
    manifest = {
        "exporter": f"coreai-fabric-llm-export {__version__}",
        "stack": _stack_versions(),
        "kind": "llm-static-graph",
        "name": args.output_name,
        "assets": {"main": f"{args.output_name}.aimodel"},
        "static_seq_len": args.seq_len,
        "compute_precision": args.compute_precision,
        "compression": None,
        "source": {"hf_model_id": args.model, "revision": args.revision},
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "Static single-graph export (no KV cache). Suitable for parity "
            "verification and batch scoring, not the Apple coreai-models chat "
            "asset layout."
        ),
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[fabric] done: {bundle_dir}")
    return bundle_dir


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
