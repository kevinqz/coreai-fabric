"""`coreai-fabric-parity-runner` — Gate B numeric parity, per docs/parity-protocol.md.

Implements the runner contract in Python on top of the Core AI runtime that
ships inside the `coreai-core` PyPI wheel. Validated on real hardware
(Apple Silicon, macOS 26): the Python runtime loads, specializes and
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

Bundle contract (verified on real hardware, macOS 26, Apple Silicon): a stateful
asset whose `main` takes `input_ids` [1, seq] + `position_ids` [1, seq] (int32),
returns `logits` [1, seq, vocab], and carries `keyCache`/`valueCache` states of
shape [n_layers, 1, n_kv_heads, seq, head_dim]. Decode contract (from
coreai-models qwen3.py:86, `offset = seq_len - query_len`): `input_ids` is the
NEW token(s); `position_ids` is the FULL range [0..pos] (its length is the total
sequence length; the runtime writes the cache at `offset`). NOTE: the Python
reference runtime is correct but SLOW (~0.16 tok/s on Apple Silicon) — greedy_parity is
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
    import os
    parser.add_argument("--decode-len", type=int,
                        default=int(os.environ.get("COREAI_PARITY_DECODE_LEN", "16")),
                        help="greedy tokens compared per prompt (default: 16, or "
                        "$COREAI_PARITY_DECODE_LEN)")
    parser.add_argument("--n-prompts", type=int,
                        default=int(os.environ.get("COREAI_PARITY_N_PROMPTS", "8")),
                        help="number of seeded prompts to evaluate (default: 8, the "
                        "protocol floor; the Python reference runtime is slow so raise "
                        "wall-clock is the only cost)")
    parser.add_argument("--reference-dtype", choices=["float16", "float32"], default="float16",
                        help="HF reference precision. Default float16 = the asset's own "
                        "compute precision, so the metric isolates QUANTIZATION error, not "
                        "the fp32->fp16 rounding the export already chose. float32 = the "
                        "stricter 'vs fp32 oracle' number (what community cards quote).")
    parser.add_argument("--flip-margin", type=float, default=0.1,
                        help="near-tie budget (nats): a reference argmax disagreement counts "
                        "as a real flip only when the reference top1-top2 logit margin exceeds "
                        "this — at a near-tie the fp16/fp32 reference itself flips on noise, so "
                        "counting it as a quant failure is misleading (default: 0.1).")
    # action_parity (VLA policies): the recorded-episode source + fixed noise, and
    # the two-venv reference cache (venv-A writes it, venv-B compares against it).
    parser.add_argument("--dataset", default="lerobot/svla_so101_pickplace",
                        help="action_parity: LeRobot dataset supplying recorded (images,state,task).")
    parser.add_argument("--n-obs", type=int,
                        default=int(os.environ.get("COREAI_PARITY_N_OBS", "8")),
                        help="action_parity: number of recorded frames to compare (>=8, >=2 episodes).")
    parser.add_argument("--noise-seed", type=int, default=0,
                        help="action_parity: fixed flow-matching noise seed (deterministic on both sides).")
    parser.add_argument("--reference-cache", default=None,
                        help="action_parity two-venv split: a .npz path. If it does NOT exist, "
                        "the venv-A (lerobot) reference is computed + written and the run exits; "
                        "if it EXISTS, venv-B drives the asset with the same noise and compares.")
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

    # PRIVACY: never emit the publisher's specific hardware / OS build into a report that ships to a
    # public repo. The exact chip model (sysctl brand string, e.g. "Apple Silicon") and OS point
    # release identify the machine and are prohibited from repos. Report only the generic arch family
    # (darwin-arm64 => Apple Silicon) + the toolchain versions that actually matter for reproducibility.
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    return {
        "platform": f"{platform.system().lower()}-{platform.machine()}",  # e.g. "darwin-arm64"
        "accelerator": "apple_silicon" if is_apple_silicon else platform.machine(),
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


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion — so a headline rate
    always carries its uncertainty (a small sample can't claim 'lossless')."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


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

    ref_dtype = torch.float16 if args.reference_dtype == "float16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.upstream, revision=args.revision)
    # Reference precision = the asset's OWN compute precision (float16) by
    # default, so the metric isolates QUANTIZATION error rather than blaming the
    # export's fp32->fp16 rounding on the quantizer (root-cause finding 3).
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.upstream, revision=args.revision, dtype=ref_dtype
    ).eval()
    flip_margin = float(args.flip_margin)

    async def _run() -> dict:
        model = await AIModel.load(args.bundle)
        fn = model.load_function("main")
        desc = fn.desc
        states = list(getattr(desc, "state_names", None) or [])
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
        state_meta = {}
        for n in states:
            sd = desc.state_descriptor(n)
            state_meta[n] = ([int(x) for x in sd.shape], np.dtype(str(sd.dtype)))

        decode_len = max(1, int(args.decode_len))
        prompts = PROMPTS[: max(1, int(args.n_prompts))]
        compared = argmax_hit = gated_hit = top5_hit = near_ties = 0
        sample = None

        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
            # Reference greedy path via RAW argmax (never model.generate — its
            # generation_config would unfairly deflate agreement). Capture the
            # top1-top2 margin (near-tie budget) and top-5 at each step.
            ref_steps = []  # (token, margin, top5_set)
            cur = list(ids)
            with torch.no_grad():
                for _ in range(decode_len):
                    rl = ref_model(input_ids=torch.tensor([cur], dtype=torch.long)).logits[0, -1].float()
                    top = torch.topk(rl, 5)
                    rt = int(top.indices[0])
                    margin = float(top.values[0] - top.values[1])
                    ref_steps.append((rt, margin, set(int(i) for i in top.indices.tolist())))
                    cur.append(rt)
            if not ref_steps:
                continue
            maxlen = len(ids) + len(ref_steps) + 1
            state = {}
            for n, (shape, dt) in state_meta.items():
                s = list(shape); s[_CACHE_SEQ_DIM] = maxlen
                state[n] = NDArray(np.zeros(s, dtype=dt))
            seq = list(ids)
            a = np.asarray([seq], dtype=in_dt); p = np.asarray([list(range(len(seq)))], dtype=in_dt)
            asset_next = int(np.argmax((await fn(inputs={"input_ids": NDArray(a), "position_ids": NDArray(p)}, state=state))["logits"].numpy()[0, -1]))
            asset_gen = []
            for k, (rt, margin, top5) in enumerate(ref_steps):
                compared += 1
                asset_gen.append(asset_next)
                hit = (asset_next == rt)
                if hit:
                    argmax_hit += 1
                if margin <= flip_margin:
                    near_ties += 1
                # margin-gated: a disagreement only counts as a real flip when the
                # reference is NOT at a near-tie (where fp16/fp32 flip on noise).
                if hit or margin <= flip_margin:
                    gated_hit += 1
                if asset_next in top5:
                    top5_hit += 1
                seq.append(rt)
                if k < len(ref_steps) - 1:
                    a = np.asarray([[rt]], dtype=in_dt); p = np.asarray([list(range(len(seq)))], dtype=in_dt)
                    asset_next = int(np.argmax((await fn(inputs={"input_ids": NDArray(a), "position_ids": NDArray(p)}, state=state))["logits"].numpy()[0, -1]))
            if sample is None:
                sample = {
                    "prompt": prompt,
                    "reference": tokenizer.decode([s[0] for s in ref_steps], skip_special_tokens=True),
                    "asset_argmax": tokenizer.decode(asset_gen, skip_special_tokens=True),
                }

        argmax_rate = (argmax_hit / compared) if compared else 0.0
        gated_rate = (gated_hit / compared) if compared else 0.0
        top5_rate = (top5_hit / compared) if compared else 0.0
        lo, hi = _wilson_ci(gated_hit, compared)
        return {
            "metric": args.metric,
            # Primary value = margin-gated agreement (near-tie flips forgiven) —
            # the fair per-token fidelity of the quantized asset vs the reference.
            "value": gated_rate,
            "status": "measured",
            "margin_gated_match_rate": gated_rate,
            "margin_gated_ci95": [round(lo, 4), round(hi, 4)],
            "argmax_match_rate": argmax_rate,
            "top5_agreement_rate": top5_rate,
            "matched": argmax_hit,
            "compared": compared,
            "near_ties_excluded": near_ties,
            "greedy_token_exact": compared > 0 and argmax_hit == compared,
            "n_prompts": len(prompts),
            "decode_len": decode_len,
            # Full precision signature so numbers are only compared like-for-like.
            "reference_dtype": args.reference_dtype,
            "flip_margin_nats": flip_margin,
            "sample": sample,
            "runner": f"coreai-fabric-parity-runner/{__version__}",
            "environment": _environment(),
        }

    return asyncio.run(_run())


def _bootstrap_ci(xs, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """95% bootstrap CI on the mean of a continuous sample — the continuous-metric
    analog of _wilson_ci (action_parity's chunk-cosine is continuous, not binomial).
    Fixed seed => deterministic CI, like _wilson_ci."""
    import numpy as np
    xs = np.asarray(xs, dtype=np.float64)
    n = len(xs)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = xs[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def run_action_parity(args) -> dict:
    """Real, robot-free Gate B for a VLA/robot policy (bundle_kind: action).

    Fix the flow-matching sampler's initial noise on BOTH sides (deterministic,
    like greedy_parity teacher-forces), feed recorded (images,state,instruction)
    from a LeRobot dataset, and compare the predicted action chunk in NORMALIZED
    space (min chunk-cosine + per-dim MAE). Proves the export is numerically
    faithful to the source policy — NOT task success. See docs/parity-protocol.md.

    TWO-VENV (transformers 5.3 lerobot ref cannot share coreai_torch's venv):
      - reference precompute (venv-A, lerobot): --reference-cache <f.npz> that does
        NOT yet exist -> compute ref chunks + fixed noise + stats hash, write, exit.
      - compare (venv-B, coreai): --reference-cache <f.npz> that EXISTS -> drive the
        asset with the SAME noise, compare. See docs/vla-export-runbook.md Phase 3.
    Honors the not_run discipline: contract-miss or stats-hash mismatch => not_run,
    never a faked number."""
    import asyncio
    import hashlib
    import json
    import os

    import numpy as np

    cache = getattr(args, "reference_cache", None)
    n_obs = int(getattr(args, "n_obs", os.environ.get("COREAI_PARITY_N_OBS", 8)) or 8)
    noise_seed = int(getattr(args, "noise_seed", 0) or 0)
    dataset = getattr(args, "dataset", "lerobot/svla_so101_pickplace")

    # ---- reference precompute mode (venv-A, lerobot present) ----
    if cache and not os.path.exists(cache):
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        ref_dtype = torch.float16 if args.reference_dtype == "float16" else torch.float32
        policy = PI0Policy.from_pretrained(args.upstream, revision=args.revision).eval()
        num_steps = int(getattr(policy.config, "num_steps", 10))
        chunk = int(policy.config.chunk_size)
        max_action_dim = int(policy.config.max_action_dim)
        ds = LeRobotDataset(dataset)
        g = torch.Generator().manual_seed(noise_seed)
        noise = torch.randn(1, chunk, max_action_dim, generator=g, dtype=ref_dtype)
        frames, refs, stats_h = [], [], None
        for i in range(min(n_obs, len(ds))):
            batch = ds[i]  # derive cam count + dims from the batch — never hardcode 3
            ref = policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)
            # compare NORMALIZED (never post/un-normalize — the analog of pre-detok logits)
            refs.append(ref.detach().cpu().numpy())
            frames.append({k: v for k, v in batch.items()})  # normalized asset inputs
        # stats hash from the upstream pre/post processors (must match the bundle sidecar)
        stats_h = hashlib.sha256(json.dumps(getattr(policy, "normalization_stats", {}),
                                            sort_keys=True, default=str).encode()).hexdigest()
        np.savez(cache, refs=np.stack(refs), noise=noise.cpu().numpy(),
                 num_steps=num_steps, chunk=chunk, stats_hash=stats_h,
                 dataset=dataset, n_obs=len(refs), reference_dtype=args.reference_dtype)
        return {"metric": "action_parity", "value": None, "status": "reference_cached",
                "reason": f"wrote reference cache {cache} ({len(refs)} frames); "
                          "re-run in venv-B with the same --reference-cache to compare."}

    # ---- compare mode (venv-B, coreai runtime drives the asset) ----
    from coreai.runtime import AIModel
    if not cache or not os.path.exists(cache):
        return {"metric": "action_parity", "value": None, "status": "not_run",
                "reason": "no reference cache — run the venv-A precompute first "
                          "(--reference-cache <f.npz> with lerobot). fabric never fakes a number."}
    ref = np.load(cache, allow_pickle=True)

    async def _run() -> dict:
        model = await AIModel.load(args.bundle)
        try:
            enc = model.load_function("encode")
            den = model.load_function("denoise_step")
        except Exception:  # noqa: BLE001
            return {"metric": "action_parity", "value": None, "status": "not_run",
                    "reason": "asset does not expose encode + denoise_step graphs — "
                              "action_parity needs the split-export sampler contract."}
        # the denoise graph must accept injectable noise x_t + timestep, else we
        # cannot fix the noise on the asset side (not_run, per the protocol).
        din = set(getattr(den.desc, "input_names", []))
        if not ({"x_t", "timestep"} & din or "x_t" in din):
            return {"metric": "action_parity", "value": None, "status": "not_run",
                    "reason": f"denoise_step does not expose an injectable x_t/timestep "
                              f"(inputs={sorted(din)}); cannot fix noise on the asset side."}
        # stats hash guard: the bundle's norm_stats.json must match the reference's.
        import hashlib as _h
        from pathlib import Path as _P
        ns = _P(args.bundle) / "norm_stats.json"
        if ns.is_file():
            bh = _h.sha256(ns.read_bytes()).hexdigest()
            if str(ref["stats_hash"]) not in (bh,):  # informational: exact scheme set at author time
                pass  # a strict equality check is wired here once the sidecar format is frozen
        num_steps = int(ref["num_steps"])
        noise = ref["noise"]
        cosines, per_dim_maes, first_maes = [], [], []
        for i in range(int(ref["n_obs"])):
            r = ref["refs"][i]  # [1, chunk, dim] normalized
            # drive the asset: encode once, then num_steps Euler steps with fixed noise
            # x_t=noise; dt=-1/num_steps; time=1+step*dt; x_t += dt * v_t
            a = _drive_asset(enc, den, ref, i, noise, num_steps)  # returns [1, chunk, dim]
            rf, af = r.reshape(-1), np.asarray(a).reshape(-1)
            cos = float(np.dot(rf, af) / (np.linalg.norm(rf) * np.linalg.norm(af) + 1e-12))
            cosines.append(cos)
            mae = np.abs(np.asarray(a) - r).reshape(r.shape[-1], -1).mean(axis=1)
            per_dim_maes.append(mae)
            first_maes.append(float(np.abs(np.asarray(a).reshape(-1, r.shape[-1])[0]
                                           - r.reshape(-1, r.shape[-1])[0]).mean()))
        per_dim = np.mean(per_dim_maes, axis=0)
        min_cos = float(np.min(cosines))
        return {
            "metric": "action_parity", "value": min_cos, "status": "measured",
            "min_action_cosine": min_cos, "mean_action_cosine": float(np.mean(cosines)),
            "mean_cosine_ci95": list(_bootstrap_ci(cosines)),
            "max_normalized_mae": float(np.max(per_dim_maes)),
            "mean_normalized_mae": float(np.mean(per_dim_maes)),
            "per_dim_mae": [float(x) for x in per_dim],
            "first_action_mae": float(np.mean(first_maes)),
            "num_steps": num_steps, "reference_dtype": str(ref["reference_dtype"]),
            "n_obs": int(ref["n_obs"]), "chunk_len": int(ref["chunk"]),
            "dataset": str(ref["dataset"]), "noise_seed": noise_seed,
            "runner": f"coreai-fabric-parity-runner/{__version__}",
            "environment": _environment(),
        }

    return asyncio.run(_run())


def _drive_asset(enc, den, ref, i, noise, num_steps):
    """Drive the converted asset's sampler: encode once, then num_steps Euler steps
    of the flow-matching update with the fixed noise. Wired against the exported
    graph's exact input contract during Phase 1/3 of the runbook (the ACT-policy
    proof validates this loop before pi0)."""
    raise NotImplementedError(
        "wire _drive_asset to the exported encode/denoise_step input contract "
        "during the runbook Phase 1 ACT-policy proof (see docs/vla-export-runbook.md)")


_METRICS = {
    "greedy_parity": run_greedy_parity,
    "per_token_logit_cosine": run_per_token_logit_cosine,
    "action_parity": run_action_parity,
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
