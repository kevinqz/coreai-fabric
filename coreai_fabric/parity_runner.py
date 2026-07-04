"""`coreai-fabric-parity-runner` — Gate B numeric parity, per docs/parity-protocol.md.

Implements the runner contract in Python on top of the Core AI runtime that
ships inside the `coreai-core` PyPI wheel. Validated on real hardware
(Apple M4 Max, macOS 26.6): the Python runtime loads, specializes and
executes `.aimodel` assets — no Swift runner and no macOS 27 required for
Gate B (macOS 27 remains the minimum for on-device deployment via the
apple/coreai-models Swift runners).

Supported metrics:
- `greedy_parity` (production stateful LLM assets) — the metric the community
  reports ("X/Y token-exact vs fp32"). Drives the real KV-cache decode of the
  `coreai.llm.export` asset and, teacher-forced along the fp32 reference's
  greedy path, measures per-token argmax agreement. GENERAL across LLM assets:
  all dims are read from the descriptor and the contract
  (input_ids+position_ids -> logits, keyCache/valueCache) is the shared one in
  coreai-models' attention. If an asset does not match that contract, the runner
  reports `not_run` — never a fake number.
- `per_token_logit_cosine` (static-graph LLM exports; greedy_token_exact).

Bundle contract (verified on real hardware, macOS 26.6, M4 Max): a stateful
asset whose `main` takes `input_ids` [1, seq] + `position_ids` [1, seq] (int32),
returns `logits` [1, seq, vocab], and carries `keyCache`/`valueCache` states of
shape [n_layers, 1, n_kv_heads, seq, head_dim]. Decode contract (from
coreai-models qwen3.py:86, `offset = seq_len - query_len`): `input_ids` is the
NEW token(s); `position_ids` is the FULL range [0..pos] (its length is the total
sequence length; the runtime writes the cache at `offset`). NOTE: the Python
reference runtime is correct but SLOW (~0.16 tok/s on M4 Max) — greedy_parity is
an opt-in local verification, not a fast/CI check. Real tok/s needs the on-device
Swift runtime (macOS 27).
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
    parser.add_argument("--decode-len", type=int, default=16,
                        help="greedy tokens compared per prompt (default: 16)")
    parser.add_argument("--n-prompts", type=int, default=4,
                        help="number of seeded prompts to evaluate (default: 4; the "
                        "Python reference runtime is slow, so this bounds wall-clock)")
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


#: The cache seq-length axis (coreai_models KVCache.seq_len_dim()). The state
#: shape is [n_layers, 1, n_kv_heads, seq, head_dim]; we resize this axis.
_CACHE_SEQ_DIM = 3


def run_greedy_parity(args) -> dict:
    """Real Gate B for a production stateful LLM asset: drive its KV-cache decode
    and, teacher-forced along the fp32 reference's greedy path, measure per-token
    argmax agreement ("X/Y token-exact vs fp32"). General/descriptor-driven; if
    the asset is not the drivable stateful-LLM contract, reports not_run."""
    import asyncio

    import numpy as np
    import torch
    from coreai.runtime import AIModel, NDArray
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.upstream, revision=args.revision)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.upstream, revision=args.revision, dtype=torch.float32
    ).eval()

    async def _run() -> dict:
        model = await AIModel.load(args.bundle)
        fn = model.load_function("main")
        desc = fn.desc
        states = list(getattr(desc, "state_names", None) or [])
        # Contract detection — decline (never fake) if it's not the drivable
        # stateful-LLM layout.
        if not ({"input_ids", "position_ids"} <= set(desc.input_names)
                and "logits" in desc.output_names and states):
            return {
                "metric": args.metric, "value": None, "status": "not_run",
                "reason": (
                    f"asset is not the drivable stateful-LLM contract "
                    f"(inputs={list(desc.input_names)}, outputs={list(desc.output_names)}, "
                    f"states={states}). greedy_parity drives input_ids+position_ids"
                    "->logits with keyCache/valueCache."),
            }
        in_dt = np.dtype(str(desc.input_descriptor("input_ids").dtype))
        # State dims straight from the descriptor (general, not hardcoded).
        state_meta = {}
        for n in states:
            sd = desc.state_descriptor(n)
            state_meta[n] = ([int(x) for x in sd.shape], np.dtype(str(sd.dtype)))

        decode_len = max(1, int(args.decode_len))
        prompts = PROMPTS[: max(1, int(args.n_prompts))]
        matched = compared = 0
        sample = None

        async def argmax_after(ids_seq, state):
            """Prefill `ids_seq` (full) and return the argmax of the last logit."""
            a = np.asarray([ids_seq], dtype=in_dt)
            p = np.asarray([list(range(len(ids_seq)))], dtype=in_dt)
            out = await fn(inputs={"input_ids": NDArray(a), "position_ids": NDArray(p)}, state=state)
            return int(np.argmax(out["logits"].numpy()[0, -1]))

        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
            # fp32 reference greedy continuation via RAW argmax — deliberately
            # NOT model.generate(), whose generation_config (repetition_penalty,
            # etc.) would penalize the reference path but not the asset's raw
            # argmax, unfairly deflating agreement. Both sides use raw argmax.
            ref_tokens = []
            cur = list(ids)
            with torch.no_grad():
                for _ in range(decode_len):
                    rl = ref_model(input_ids=torch.tensor([cur], dtype=torch.long)).logits[0, -1]
                    rt = int(torch.argmax(rl))
                    ref_tokens.append(rt)
                    cur.append(rt)
            if not ref_tokens:
                continue
            # Fresh KV cache sized to the full teacher sequence.
            maxlen = len(ids) + len(ref_tokens) + 1
            state = {}
            for n, (shape, dt) in state_meta.items():
                s = list(shape); s[_CACHE_SEQ_DIM] = maxlen
                state[n] = NDArray(np.zeros(s, dtype=dt))
            # Prefill the prompt, then teacher-force the reference tokens,
            # checking the asset's argmax against each reference next-token.
            seq = list(ids)
            asset_next = await argmax_after(seq, state)
            asset_gen = []
            for k, rt in enumerate(ref_tokens):
                compared += 1
                asset_gen.append(asset_next)
                if asset_next == rt:
                    matched += 1
                seq.append(rt)  # advance along the REFERENCE path (aligned)
                if k < len(ref_tokens) - 1:
                    # incremental: feed the new (reference) token; position_ids is
                    # the FULL range so offset = len-1 writes at the right slot.
                    a = np.asarray([[rt]], dtype=in_dt)
                    p = np.asarray([list(range(len(seq)))], dtype=in_dt)
                    out = await fn(inputs={"input_ids": NDArray(a), "position_ids": NDArray(p)}, state=state)
                    asset_next = int(np.argmax(out["logits"].numpy()[0, -1]))
            if sample is None:
                sample = {
                    "prompt": prompt,
                    "reference": tokenizer.decode(ref_tokens, skip_special_tokens=True),
                    "asset_argmax": tokenizer.decode(asset_gen, skip_special_tokens=True),
                }

        rate = (matched / compared) if compared else 0.0
        return {
            "metric": args.metric,
            "value": rate,
            "status": "measured",
            "matched": matched,
            "compared": compared,
            "match_rate": rate,
            "greedy_token_exact": compared > 0 and matched == compared,
            "n_prompts": len(prompts),
            "decode_len": decode_len,
            "sample": sample,
            "reference_dtype": "float32",
            "runner": f"coreai-fabric-parity-runner/{__version__}",
            "environment": _environment(),
        }

    return asyncio.run(_run())


_METRICS = {
    "greedy_parity": run_greedy_parity,
    "per_token_logit_cosine": run_per_token_logit_cosine,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner = _METRICS.get(args.metric)
    if runner is None:
        print(
            f"error: metric {args.metric!r} is not implemented by "
            f"coreai-fabric-parity-runner {__version__} (supported: "
            f"{', '.join(sorted(_METRICS))}). Gate B stays honest: this run fails "
            "rather than fakes a result.",
            file=sys.stderr,
        )
        return 3

    try:
        report = runner(args)
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
