"""`coreai-fabric-parity-runner` — Gate B numeric parity, per docs/parity-protocol.md.

Implements the runner contract in Python on top of the Core AI runtime that
ships inside the `coreai-core` PyPI wheel. Validated on real hardware
(Apple M4 Max, macOS 26.6): the Python runtime loads, specializes and
executes `.aimodel` assets — no Swift runner and no macOS 27 required for
Gate B (macOS 27 remains the minimum for on-device deployment via the
apple/coreai-models Swift runners).

Supported metrics (v0.1):
- per_token_logit_cosine (LLMs; greedy_token_exact supported)

`graph_output_cosine` needs per-modality reference preprocessing and is not
implemented yet — the runner exits non-zero with an honest message, which
verify records as a Gate B failure rather than a fake pass.

Bundle expectations (the layout produced by coreai-fabric-llm-export and
verified against real assets): `<bundle>.aimodel` with a `main` function
taking a static (1, seq_len) `input_ids` tensor and returning full-sequence
`logits`.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys

__version__ = "0.1.0"

#: Deterministic prompt corpus (protocol: seeded, N >= 8). Versioned here so
#: runs are reproducible; changing it is a protocol-relevant change.
PROMPTS = [
    "The capital of France is",
    "In a shocking finding, scientists discovered",
    "def fibonacci(n):",
    "The three primary colors are",
    "Once upon a time, in a village by the sea,",
    "The chemical symbol for gold is",
    "To be, or not to be, that is",
    "Water boils at a temperature of",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coreai-fabric-parity-runner")
    parser.add_argument("--bundle", required=True, help="path to <id>.aimodel")
    parser.add_argument("--upstream", required=True, help="upstream HF repo (owner/name)")
    parser.add_argument("--metric", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--tolerance", type=float, required=True)
    parser.add_argument("--report-json", default="-", help="'-' writes the report to stdout")
    parser.add_argument("--revision", default=None, help="upstream HF revision to pin (optional)")
    parser.add_argument("--decode-len", type=int, default=64,
                        help="greedy decode steps per prompt (protocol default: 64; "
                        "clamped to the bundle's static seq len and recorded)")
    parser.add_argument("--version", action="version",
                        version=f"coreai-fabric-parity-runner {__version__}")
    return parser


def _chip() -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def _environment() -> dict:
    from importlib import metadata

    def v(dist):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            return None

    return {
        "os": f"macOS {platform.mac_ver()[0]}" if platform.mac_ver()[0] else platform.platform(),
        "chip": _chip(),
        "machine": platform.machine(),
        "runtime_version": v("coreai-core"),
        "coreai_torch": v("coreai-torch"),
        "torch": v("torch"),
        "transformers": v("transformers"),
    }


def run_per_token_logit_cosine(args) -> dict:
    import asyncio

    import numpy as np
    import torch
    from coreai.runtime import AIModel, NDArray
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.upstream, revision=args.revision)
    # Reference runs in float32: the upstream model as published, at full
    # precision. Divergence therefore includes the recipe's precision choice —
    # recorded, not hidden.
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.upstream, revision=args.revision, dtype=torch.float32
    )
    ref_model.eval()

    async def _run() -> dict:
        model = await AIModel.load(args.bundle)
        fn = model.load_function("main")
        # Read the static shape + dtype from the function descriptor (verified
        # API: desc.input_descriptor(name) -> NDArrayDescriptor with
        # .shape/.dtype; coreai-torch narrows torch int64 ids to int32).
        desc = fn.desc
        # A production `coreai.llm.export` asset is STATEFUL: it carries a KV
        # cache (state_names = keyCache/valueCache) and takes input_ids +
        # position_ids with a dynamic seq len. This static-graph runner cannot
        # drive it (a plain forward raises "Missing state view for keyCache"),
        # and raw logit-cosine vs an fp32 reference is the wrong metric for a
        # quantized asset anyway (4bit diverges in logits but preserves task
        # accuracy). The correct Gate B is a benchmark-accuracy eval via Apple's
        # `coreai.llm.eval` — which ships as a STUB ("Evaluation support is
        # coming soon") as of coreai-models 0.1.0. So we honestly decline rather
        # than fake a number. Verified on real hardware 2026-07-03.
        if getattr(desc, "state_names", None):
            return {
                "metric": args.metric,
                "value": None,
                "status": "not_run",
                "reason": (
                    "production KV-cache asset (state_names="
                    f"{list(desc.state_names)}): this static-graph runner cannot "
                    "drive a stateful asset, and raw logit-cosine is the wrong "
                    "metric for a quantized asset. Gate B for production assets "
                    "is benchmark-accuracy via Apple's coreai.llm.eval, which is "
                    "not yet implemented upstream (coreai-models 0.1.0: "
                    "'Evaluation support is coming soon')."
                ),
            }
        if "input_ids" not in desc.input_names or "logits" not in desc.output_names:
            raise SystemExit(
                f"bundle main function has inputs {desc.input_names} / outputs "
                f"{desc.output_names}; expected input_ids -> logits "
                "(the coreai-fabric-llm-export layout)"
            )
        in_desc = desc.input_descriptor("input_ids")
        shape = list(in_desc.shape)
        if len(shape) != 2:
            raise SystemExit(f"input_ids has unexpected shape {shape}")
        seq_len = int(shape[1])
        in_dtype = np.dtype(str(in_desc.dtype))

        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        min_cos = 1.0
        greedy_exact = True
        decode_len_effective = None

        for prompt in PROMPTS:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
            steps = min(args.decode_len, seq_len - len(ids))
            decode_len_effective = steps if decode_len_effective is None else min(
                decode_len_effective, steps
            )
            for _ in range(steps):
                pos = len(ids) - 1
                padded = ids + [pad_id] * (seq_len - len(ids))
                arr = np.asarray([padded], dtype=in_dtype)
                out = await fn(inputs={"input_ids": NDArray(arr)})
                got = out["logits"].numpy()[0, pos].astype(np.float64)

                with torch.no_grad():
                    ref = ref_model(
                        input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False
                    ).logits[0, -1].double().numpy()

                cos = float(
                    np.dot(ref, got) / (np.linalg.norm(ref) * np.linalg.norm(got))
                )
                min_cos = min(min_cos, cos)
                ref_tok = int(np.argmax(ref))
                got_tok = int(np.argmax(got))
                if ref_tok != got_tok:
                    greedy_exact = False
                # Continue along the REFERENCE greedy path so both sides stay
                # aligned even after a mismatch.
                ids.append(ref_tok)

        return {
            "value": min_cos,
            "greedy_token_exact": greedy_exact,
            "n_inputs": len(PROMPTS),
            "decode_len": decode_len_effective,
            "static_seq_len": seq_len,
            "reference_dtype": "float32",
            "runner": f"coreai-fabric-parity-runner/{__version__}",
            "environment": _environment(),
        }

    return asyncio.run(_run())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.metric != "per_token_logit_cosine":
        print(
            f"error: metric {args.metric!r} is not implemented by "
            f"coreai-fabric-parity-runner {__version__} (supported: "
            "per_token_logit_cosine). Gate B stays honest: this run fails "
            "rather than fakes a result.",
            file=sys.stderr,
        )
        return 3

    try:
        report = run_per_token_logit_cosine(args)
    except ImportError as exc:
        print(
            f"error: missing runtime stack: {exc}\n"
            'install with: pip install "coreai-fabric[convert]"',
            file=sys.stderr,
        )
        return 1

    payload = json.dumps(report, indent=2)
    if args.report_json == "-":
        print(payload)
    else:
        with open(args.report_json, "w") as fh:
            fh.write(payload + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
