"""pi0fast AUTOREGRESSIVE greedy-token parity — the AR analogue of the flow-matching VLA parities.

pi0/pi05/smolvla are flow-matching (encode -> Euler denoise loop); their parity is action-chunk
cosine. pi0fast is AUTOREGRESSIVE: encode -> greedy decode loop over discrete FAST action tokens ->
(host) FAST-detokenize -> continuous actions. The action tokens are the model's entire output; the
detokenizer is a fixed deterministic host transform, so **matching the greedy token stream is a
complete fidelity statement** (identical tokens => identical actions, by construction). We therefore
measure greedy-TOKEN parity — the same principle as the LLM lane's `greedy_parity`.

No tokenizer is needed: `sample_actions_fast` consumes only `bos_token_id` (=2, the _StubTok), and the
existing pi0/pi05/smolvla parities already use fixed synthetic language tokens rather than a real
prompt. (The PaliGemma text tokenizer google/paligemma-3b-pt-224 is GATED and this account is not
authorized — but it is only needed host-side to detokenize tokens into robot actions, never to prove
export fidelity. Detokenization is out of scope for the fidelity metric.)

TWO METRICS (both over a fixed decode length, synthetic distinct [-1,1] images + fixed language tokens):
  - teacher_forced_token_parity (PRIMARY): at each step feed the TRUE reference token history to the
      coreai decode_step and check its argmax == the reference's argmax. Isolates per-step lowering
      fidelity from autoregressive cascade. This is the gate `value`.
  - free_running_first_divergence: coreai decodes from its OWN tokens; report the step where it first
      diverges from the reference (how long the two stay bit-locked). Informative, not the gate.

TWO-VENV:
  reference (venv-A, .venv-lerobot): real PI0FastPolicy (+ bos stub), sample_actions_fast (RECOMPUTE —
      the exact math the graph implements) -> ref tokens. Saves obs + ref tokens.
  --compare (venv-B, .venv + coreai): lowered asset. encode -> prefix_embeds; host recompute loop over
      decode_step -> teacher-forced argmaxes + free-running tokens. Reports the two metrics.

  .venv-lerobot/bin/python models/pi0fast/parity.py reference --out build/pi0fast-base
  .venv/bin/python         models/pi0fast/parity.py --compare  --out build/pi0fast-base \
        --bundle build/pi0fast-base/pi0fast-base.aimodel
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export as pf  # noqa: E402 — constants (PREFIX_LEN, MAX_TOTAL, MAX_DECODE, VOCAB) + natural_image

REF_NPZ = "pi0fast_parity_ref.npz"
MASK_VALUE = -2.3819763e38  # OPENPI_ATTENTION_MASK_VALUE (-> -inf in fp16)


def _synthetic_lang_tokens(g, tok_len, vocab=257152):
    """Fixed, diverse, valid language-token ids (no gated tokenizer). Diversity (vs all-zero) gives a
    non-degenerate reference generation, so the token-parity test actually exercises the decoder."""
    import torch
    return (torch.randint(4, min(vocab, 100000), (1, tok_len), generator=g),
            torch.ones(1, tok_len, dtype=torch.bool))


def _obs(seed, tok_len, fp):
    import torch
    g = torch.Generator().manual_seed(seed)
    imgs = [pf.natural_image(g).to(fp) for _ in range(3)]     # [-1,1] deployment/SigLIP range
    imasks = [torch.ones(1, dtype=torch.bool) for _ in range(3)]
    ltok, lmask = _synthetic_lang_tokens(g, tok_len)
    return imgs, imasks, ltok, lmask


def cmd_reference(out: Path, n_frames: int, seed: int, max_decode: int, fp16=True):
    import numpy as np
    import torch
    policy = pf._load_policy(fp16=fp16)          # real weights (PI0_CONFIG_DIR) + bos stub
    fd = torch.float16 if fp16 else torch.float32
    imgs_all, ltok_all, lmask_all, reftok_all = [], [], [], []
    for i in range(n_frames):
        imgs, imasks, ltok, lmask = _obs(seed + i, pf.TOK, fd)
        with torch.no_grad():
            # KV-cache generation (O(n)) yields the IDENTICAL greedy token sequence as the O(n^2)
            # recompute `sample_actions_fast` — the cache is exact for causal attention — but is ~10x
            # faster (fp16 CPU recompute is intractable). The coreai side does recompute; the
            # teacher-forced metric checks coreai's per-step argmax against these reference tokens, so
            # it is invariant to how the reference history was generated.
            ref_tok = policy.model.sample_actions_fast_kv_cache(
                imgs, imasks, ltok, lmask, max_decoding_steps=max_decode, temperature=0.0)
        rt = ref_tok[0].cpu().numpy().astype(np.int64)
        uniq = len(np.unique(rt))
        print(f"  obs {i}: generated {len(rt)} tokens, {uniq} unique (diversity check)")
        imgs_all.append(np.stack([x.float().numpy()[0] for x in imgs]))   # [3,3,224,224]
        ltok_all.append(ltok.cpu().numpy()[0])
        lmask_all.append(lmask.cpu().numpy()[0])
        reftok_all.append(rt)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / REF_NPZ,
        images=np.stack(imgs_all).astype(np.float32),
        lang_tokens=np.stack(ltok_all).astype(np.int64),
        lang_masks=np.stack(lmask_all).astype(bool),
        ref_tokens=np.stack(reftok_all).astype(np.int64),
        max_decode=max_decode, prefix_len=pf.PREFIX_LEN, max_total=pf.MAX_TOTAL,
    )
    print(f"ok: wrote {out/REF_NPZ}  (n={n_frames}, max_decode={max_decode})")
    print("next (venv-B): export --lower, then this script --compare")


def cmd_compare(out: Path, bundle: Path, max_steps: int = 10 ** 9, compute_unit: str = "cpu",
                free_running: bool = True):
    import asyncio
    import json
    import numpy as np
    from coreai.runtime import AIModel, NDArray, ComputeUnitKind, SpecializationOptions
    from coreai_fabric import __version__
    from coreai_fabric.parity_runner import _environment

    d = np.load(out / REF_NPZ, allow_pickle=True)
    images, ltoks, lmasks, ref_tokens = d["images"], d["lang_tokens"], d["lang_masks"], d["ref_tokens"]
    MD_ref = int(d["max_decode"])
    PL, MT, MD = int(d["prefix_len"]), int(d["max_total"]), int(pf.MAX_DECODE)

    async def _run() -> dict:
        _cu = {"cpu": ComputeUnitKind.cpu, "gpu": ComputeUnitKind.gpu,
               "neural_engine": ComputeUnitKind.neural_engine}[compute_unit]
        model = await AIModel.load(str(bundle),
                                   SpecializationOptions.from_preferred_compute_unit_kind(_cu()))
        try:
            enc = model.load_function("encode")
            dec = model.load_function("decode_step")
        except Exception:  # noqa: BLE001
            return {"metric": "action_parity", "value": None, "status": "not_run",
                    "reason": "asset lacks encode + decode_step entrypoints"}

        def nd(fn, name, arr):
            dt = np.dtype(str(fn.desc.input_descriptor(name).dtype))
            return NDArray(np.asarray(arr).astype(dt))

        CACHE_NAMES = pf._CACHE_K + pf._CACHE_V           # 36 KV tensors [1,KV_HEADS,MAX_TOTAL,HEAD_DIM]

        async def do_encode(i):
            """encode = SigLIP x3 + prefix prefill -> first token (argmax of prefill logits) + KV cache."""
            eo = await enc(inputs={
                "img0": nd(enc, "img0", images[i][0:1]), "img1": nd(enc, "img1", images[i][1:2]),
                "img2": nd(enc, "img2", images[i][2:3]), "lang_tokens": nd(enc, "lang_tokens", ltoks[i:i + 1])})
            return int(np.argmax(eo["first_logits"].numpy()[0])), {n: eo[n].numpy() for n in CACHE_NAMES}

        async def decode_one(cache, tok_in, wp):
            """feed tok_in at cache position wp (write its K/V there); attend to [0:wp+1] -> next argmax."""
            att = np.zeros((1, 1, 1, MT), np.float32)
            att[:, :, :, wp + 1:] = MASK_VALUE
            din = {"new_token": nd(dec, "new_token", np.array([[tok_in]], np.int64)),
                   "att_4d": nd(dec, "att_4d", att),
                   "position_id": nd(dec, "position_id", np.array([[wp]], np.int64)),
                   "cache_position": nd(dec, "cache_position", np.array([wp], np.int64))}
            din.update({n: nd(dec, n, cache[n]) for n in CACHE_NAMES})
            do = await dec(inputs=din)
            return int(np.argmax(do["logits"].numpy()[0])), {n: do[n].numpy() for n in CACHE_NAMES}

        rows = []
        for i in range(len(images)):
            first, base_cache = await do_encode(i)        # gen_0 = argmax of the prefill; cache filled [0:PL]
            print(f"  obs {i}: encoded (first token), decoding...", flush=True)
            rt = ref_tokens[i]
            n = min(MD_ref, MD, max_steps)
            # teacher-forced: gen_0 from the prefill; then feed the TRUE prior token gen_{t-1} at position
            # PL+(t-1) each step and check the argmax against the reference gen_t.
            tf_match = np.zeros(n, bool)
            tf_match[0] = (first == int(rt[0]))
            cache = {k: v.copy() for k, v in base_cache.items()}
            for t in range(1, n):
                wp = PL + (t - 1)
                pred, cache = await decode_one(cache, int(rt[t - 1]), wp)
                tf_match[t] = (pred == int(rt[t]))
                if (t + 1) % 8 == 0:
                    print(f"    tf step {t + 1}/{n}  running_parity={tf_match[:t + 1].mean():.3f}", flush=True)
            # free-running: decode from the model's OWN tokens (fresh cache copy); report first divergence
            first_div = -1
            if free_running:
                first_div = n
                if first != int(rt[0]):
                    first_div = 0
                else:
                    cache = {k: v.copy() for k, v in base_cache.items()}
                    prev = first
                    for t in range(1, n):
                        pred, cache = await decode_one(cache, prev, PL + (t - 1))
                        if pred != int(rt[t]):
                            first_div = t
                            break
                        prev = pred
            rows.append({"obs": i, "n_steps": n,
                         "teacher_forced_parity": float(tf_match.mean()),
                         "tf_first_mismatch": int(np.argmax(~tf_match)) if not tf_match.all() else n,
                         "free_running_first_divergence": int(first_div)})
            print(f"  obs {i}: tf_parity={tf_match.mean():.4f} "
                  f"tf_first_mismatch={rows[-1]['tf_first_mismatch']}/{n} "
                  f"free_run_first_div={first_div}/{n}", flush=True)

        tf = np.array([r["teacher_forced_parity"] for r in rows])
        fdiv = np.array([r["free_running_first_divergence"] for r in rows]) if free_running else None
        return {
            "metric": "action_parity", "value": float(tf.min()), "status": "measured",
            "parity_kind": "autoregressive_greedy_token", "compute_unit": compute_unit,
            "min_teacher_forced_token_parity": float(tf.min()),
            "mean_teacher_forced_token_parity": float(tf.mean()),
            "min_free_running_first_divergence": int(fdiv.min()) if free_running else None,
            "mean_free_running_first_divergence": float(fdiv.mean()) if free_running else None,
            "per_obs": rows, "n_obs": int(len(images)), "decode_steps": int(min(MD_ref, MD, max_steps)),
            "sampler": "autoregressive_greedy", "deterministic": True,
            "observations": "synthetic distinct 1/f natural images ([-1,1]) + fixed language tokens — "
                            "conversion-fidelity metric (coreai vs torch fp16, identical inputs)",
            "reference": "torch fp16 sample_actions_fast (RECOMPUTE, not kv-cache) vs the fp16 "
                         "coreai-optimized asset over the identical host recompute loop. Metric is "
                         "greedy action-TOKEN parity: the FAST tokens are the model's full output and "
                         "detokenization is a fixed host transform, so matching tokens => matching "
                         "actions. teacher_forced isolates per-step fidelity; free_running reports "
                         "cascade lock length.",
            "runner": f"coreai-fabric-parity-runner/{__version__}", "environment": _environment(),
        }

    result = asyncio.run(_run())
    emit = out / "action-parity-measured.json"
    emit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nok: wrote {emit} (feeds `coreai-fabric verify {out.name}`)")


def main():
    ap = argparse.ArgumentParser(description="pi0fast autoregressive greedy-token parity")
    ap.add_argument("phase", nargs="?", default="reference", choices=["reference"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--n-frames", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-decode", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=10 ** 9,
                    help="cap the compare decode loop (the coreai recompute is O(n^2)/step, ~1min/token)")
    ap.add_argument("--compute-unit", default="cpu", choices=["cpu", "gpu", "neural_engine"])
    ap.add_argument("--no-free", action="store_true", help="skip the free-running loop (halves forwards)")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()
    if args.compare:
        cmd_compare(args.out, args.bundle or (args.out / f"{args.out.name}.aimodel"),
                    max_steps=args.max_steps, compute_unit=args.compute_unit, free_running=not args.no_free)
    else:
        cmd_reference(args.out, args.n_frames, args.seed, args.max_decode, fp16=not args.fp32)


if __name__ == "__main__":
    main()
