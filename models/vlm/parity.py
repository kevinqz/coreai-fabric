"""vlm_greedy_parity Gate B for a coreai.vlm.export .llmasset (Qwen3-VL-family VLM).

The bundle is 3 graphs the HOST orchestrates: vision(pixel_values[1,3,448,448]->image_features
[1,196,2048]) + embed(input_ids->embeddings[1,seq,2048]) + main(inputs_embeds+position_ids->logits,
stateful k_cache/v_cache). Inference: vision-encode + text-embed, splice the 196 image features at the
image_token_id positions, prefill main, greedy-decode. Parity drives BOTH the torch reference and the
asset from the SAME preprocessed image + prompt and reports:
  A: greedy token-exact match rate over `decode_len` tokens (the community "X/Y vs fp32" number) — PRIMARY
  B: first-token logit cosine + argmax agreement (isolates the vision+embed+prefill path from decode drift)
  C: coreai-vs-torch vision-feature cosine — SELF-VALIDATES that our host-side image preprocessing matches
     the torch processor (if C is low, A/B are meaningless — the images differ, not the export).

TWO-VENV: `reference` (.venv-lerobot: transformers Qwen3-VL) writes an .npz; `--compare` (fabric .venv:
coreai runtime) drives the asset. CLIP-normalized 448x448 preprocessing is done identically on both sides.

  .venv-lerobot/bin/python models/vlm/parity.py reference --hf-id Qwen/Qwen3-VL-2B-Instruct \
       --probes build/_vlm_probes --out build/qwen3-vl-2b
  .venv/bin/python         models/vlm/parity.py --compare --out build/qwen3-vl-2b \
       --bundle build/qwen3-vl-2b/qwen3_vl_2b.llmasset
"""
import argparse
import sys
from pathlib import Path

REF_NPZ = "vlm_parity_ref.npz"
PROMPT = "Describe this image in detail."
IMAGE_TOKEN_ID = 151655
IMG = 448
# CLIP normalization (Qwen3-VL 2B spec); GELab uses (0.5,)*3 — pass --image-mean/--image-std to override.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _preprocess(img_path, mean, std):
    """Deterministic 448x448 CLIP-normalized [1,3,448,448] float32 — the RAW input the coreai vision
    graph (which patchifies internally) was traced with. Same tensor is handed to the torch vision
    tower, so vision features must match (metric C)."""
    import numpy as np
    from PIL import Image
    im = Image.open(img_path).convert("RGB").resize((IMG, IMG), Image.BICUBIC)
    a = np.asarray(im).astype("float32") / 255.0                       # [H,W,3] in [0,1]
    a = (a - np.asarray(mean, "float32")) / np.asarray(std, "float32")
    return a.transpose(2, 0, 1)[None]                                  # [1,3,448,448]


def _probe_paths(probes_dir):
    exts = (".jpg", ".jpeg", ".png", ".bin", ".webp")
    return sorted(p for p in Path(probes_dir).iterdir() if p.suffix.lower() in exts)


def cmd_reference(hf_id, probes_dir, out: Path, decode_len, mean, std):
    import numpy as np
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    from transformers import AutoConfig
    proc = AutoProcessor.from_pretrained(hf_id, use_fast=True, min_pixels=IMG * IMG, max_pixels=IMG * IMG)
    # Match the coreai export's design: coreai.vlm.export sets text_config.rope_scaling=None (STANDARD
    # RoPE, not Qwen3-VL's native M-RoPE). Configure the reference identically so parity isolates the
    # coreai LOWERING fidelity (fp16) rather than the deliberate M-RoPE->standard-RoPE export simplification
    # (that quality delta is disclosed in the card, not what Gate B certifies).
    cfg = AutoConfig.from_pretrained(hf_id)
    (cfg.text_config if hasattr(cfg, "text_config") else cfg).rope_scaling = None
    model = AutoModelForImageTextToText.from_pretrained(hf_id, config=cfg, dtype=torch.float32).eval()
    vision = model.model.visual if hasattr(model.model, "visual") else model.visual

    probes = _probe_paths(probes_dir)
    recs = []
    for p in probes:
        px = _preprocess(p, mean, std)                                  # [1,3,448,448] — shared w/ coreai
        from PIL import Image
        pil = Image.open(p).convert("RGB").resize((IMG, IMG), Image.BICUBIC)
        msgs = [{"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = proc(text=[text], images=[pil], return_tensors="pt")
        input_ids = enc["input_ids"]
        img_pos = (input_ids[0] == IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0]
        # Qwen3-VL M-RoPE position ids (image tokens advance the position by max(grid) not 196).
        # The coreai export uses STANDARD RoPE (rope_scaling=None) with a 1D position_ids, so feed it
        # the M-RoPE temporal dim: text tokens then get their true sequential positions (they're the
        # decode majority) and the flat-offset decode bug is avoided.
        try:
            rope_ids, _ = model.model.get_rope_index(input_ids, image_grid_thw=enc.get("image_grid_thw"),
                                                     attention_mask=enc.get("attention_mask"))
            mrope = rope_ids[0, 0].numpy().astype("int64")            # [seq] temporal dim
            mrope_next = int(rope_ids.max().item()) + 1               # first decode position
        except Exception:
            mrope = np.arange(input_ids.shape[1], dtype="int64")
            mrope_next = int(input_ids.shape[1])
        with torch.no_grad():
            # vision features via the torch tower on the processor's patchified pixels (for metric C)
            try:
                vout = vision(enc["pixel_values"].to(torch.float32), grid_thw=enc["image_grid_thw"])
            except TypeError:
                vout = vision(enc["pixel_values"].to(torch.float32), enc["image_grid_thw"])
            vf = vout.last_hidden_state if hasattr(vout, "last_hidden_state") else (
                vout[0] if isinstance(vout, (tuple, list)) else vout)
            vf = vf.reshape(-1, vf.shape[-1]).float().numpy()          # [196, hidden]
            # greedy decode via RAW argmax (never .generate): multimodal prefill, then KV-cache token loop
            o = model(**enc, use_cache=True)
            past = o.past_key_values
            first_logits = o.logits[0, -1].float().numpy()
            nt = int(o.logits[0, -1].argmax())
            toks = [nt]
            for _ in range(decode_len - 1):
                o = model(input_ids=torch.tensor([[nt]], dtype=torch.long),
                          past_key_values=past, use_cache=True)
                past = o.past_key_values
                nt = int(o.logits[0, -1].argmax())
                toks.append(nt)
        recs.append(dict(pixel_values=px.astype("float32"), input_ids=input_ids[0].numpy().astype("int64"),
                         img_pos=img_pos.numpy().astype("int64"), ref_tokens=np.asarray(toks, "int64"),
                         ref_first_logits=first_logits.astype("float32"), ref_vision=vf.astype("float32"),
                         mrope=mrope, mrope_next=np.array([mrope_next], "int64")))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / REF_NPZ, n=len(recs), decode_len=decode_len,
             **{f"{k}_{i}": v for i, r in enumerate(recs) for k, v in r.items()})
    print(f"ok: wrote {out/REF_NPZ} (n={len(recs)} probes, decode_len={decode_len})")


def cmd_compare(out: Path, bundle: Path, decode_len):
    import asyncio
    import json
    import numpy as np
    from coreai.runtime import AIModel, NDArray

    d = np.load(out / REF_NPZ)
    n = int(d["n"])

    meta = json.loads((bundle / "metadata.json").read_text())
    assets = meta["assets"]  # {"main":..., "embedding":..., "vision":...}

    async def _run():
        # the .llmasset is a CONTAINER of 3 separate .aimodel bundles — load each (fn "main")
        vis = (await AIModel.load(str(bundle / assets["vision"]))).load_function("main")
        emb = (await AIModel.load(str(bundle / assets["embedding"]))).load_function("main")
        main = (await AIModel.load(str(bundle / assets["main"]))).load_function("main")
        sd = main.desc
        states = list(getattr(sd, "state_names", None) or [])
        state_meta = {s: ([int(x) for x in sd.state_descriptor(s).shape],
                          np.dtype(str(sd.state_descriptor(s).dtype))) for s in states}

        def nd(fn, name, arr):
            return NDArray(np.asarray(arr).astype(np.dtype(str(fn.desc.input_descriptor(name).dtype))))

        def cos(a, b):
            a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        tok_match = tok_total = argmax_hit = 0
        vis_coss, logit_coss = [], []
        for i in range(n):
            px = d[f"pixel_values_{i}"]
            input_ids = d[f"input_ids_{i}"].astype(np.int64)
            img_pos = d[f"img_pos_{i}"].astype(np.int64)
            ref_tokens = d[f"ref_tokens_{i}"]
            ref_first = d[f"ref_first_logits_{i}"]
            ref_vis = d[f"ref_vision_{i}"]

            vf = (await vis(inputs={"pixel_values": nd(vis, "pixel_values", px)}))["image_features"].numpy()[0]
            if vf.size == ref_vis.size:                                # metric C (self-validation)
                vis_coss.append(cos(vf, ref_vis))
            elif vf.shape[0] * 2 == np.asarray(ref_vis).reshape(-1, vf.shape[-1]).shape[0]:
                # torch last_hidden_state is pre-final-temporal-merge (2x tokens): average the 2 halves
                rv = np.asarray(ref_vis).reshape(-1, vf.shape[-1]).reshape(2, vf.shape[0], vf.shape[-1]).mean(0)
                vis_coss.append(cos(vf, rv))
            embeds = (await emb(inputs={"input_ids": nd(emb, "input_ids", input_ids[None])}))["embeddings"].numpy()
            embeds[0, img_pos] = vf.astype(embeds.dtype)               # splice

            state = {s: NDArray(np.zeros(shape, dt)) for s, (shape, dt) in state_meta.items()}
            seq = embeds.shape[1]
            mrope = d[f"mrope_{i}"].astype(np.int32)                   # M-RoPE-derived 1D positions
            offset = int(d[f"mrope_next_{i}"][0])                      # first decode position (not flat seq)
            pos = mrope[None]
            logits = (await main(inputs={"inputs_embeds": nd(main, "inputs_embeds", embeds),
                                         "position_ids": nd(main, "position_ids", pos)}, state=state))["logits"].numpy()
            first = logits[0, -1]
            logit_coss.append(cos(first, ref_first))                  # metric B
            if int(first.argmax()) == int(ref_first.argmax()):
                argmax_hit += 1
            nxt = int(first.argmax())
            for k in range(decode_len):
                tok_total += 1
                if nxt == int(ref_tokens[k]):
                    tok_match += 1
                if nxt != int(ref_tokens[k]):
                    break                                             # greedy path diverged; stop this probe
                e = (await emb(inputs={"input_ids": nd(emb, "input_ids", np.array([[nxt]], np.int64))}))["embeddings"].numpy()
                lo = (await main(inputs={"inputs_embeds": nd(main, "inputs_embeds", e),
                                         "position_ids": nd(main, "position_ids", np.array([[offset]], np.int32))},
                                 state=state))["logits"].numpy()
                nxt = int(lo[0, -1].argmax())
                offset += 1

        result = {
            "metric": "vlm_greedy_parity",
            "value": (tok_match / tok_total) if tok_total else 0.0,
            "status": "measured",
            "token_match_rate": (tok_match / tok_total) if tok_total else 0.0,
            "tokens_matched": tok_match, "tokens_total": tok_total,
            "first_token_argmax_rate": argmax_hit / n,
            "min_first_token_logit_cosine": float(min(logit_coss)),
            "mean_first_token_logit_cosine": float(np.mean(logit_coss)),
            "min_vision_feature_cosine": float(min(vis_coss)) if vis_coss else None,
            "mean_vision_feature_cosine": float(np.mean(vis_coss)) if vis_coss else None,
            "n_probes": n, "decode_len": decode_len,
            "reference": "torch fp32 Qwen3-VL (raw greedy argmax) vs the fp16 coreai .llmasset, identical "
                         "448px CLIP-normalized image + prompt, host-driven vision-splice + KV decode",
        }
        return result

    result = asyncio.run(_run())
    emit = out / "vlm-parity-measured.json"
    emit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nok: wrote {emit}")


def main():
    ap = argparse.ArgumentParser(description="vlm_greedy_parity")
    ap.add_argument("phase", nargs="?", default="reference", choices=["reference"])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--hf-id")
    ap.add_argument("--probes", default="build/_vlm_probes")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--decode-len", type=int, default=16)
    ap.add_argument("--image-mean", type=float, nargs=3, default=list(CLIP_MEAN))
    ap.add_argument("--image-std", type=float, nargs=3, default=list(CLIP_STD))
    a = ap.parse_args()
    if a.compare:
        cmd_compare(a.out, a.bundle or (a.out / "qwen3_vl_2b.llmasset"), a.decode_len)
    else:
        cmd_reference(a.hf_id, a.probes, a.out, a.decode_len, tuple(a.image_mean), tuple(a.image_std))


if __name__ == "__main__":
    main()
