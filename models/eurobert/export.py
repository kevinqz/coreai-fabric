# /// script
# requires-python = ">=3.12"
# ///
"""EuroBERT export lane (pulpie / token-classification and siblings).

A NON-autoregressive encoder: `(input_ids, attention_mask) -> per-token logits`.
One static-seq `.aimodel`, single entrypoint `main`. The host owns the tokenizer.
Gate B is `graph_output_cosine` (models/eurobert/parity.py) — the encoder analog
of the LLM logit-parity: identical seeded inputs through both the torch reference
and the lowered asset, compared by cosine.

EuroBERT is RoPE + GQA + bidirectional (no causal mask, no KV-cache) — op-coverage
proven on coreai_torch 0.4.1 / coremltools 9.0.

Usage (single env — transformers + coreai_torch coexist in fabric's .venv):
  .venv/bin/python models/eurobert/export.py export \
      --model build/_eurobert/pulpie-orange-base --out build/pulpie-orange-base --seq 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

DEFAULT_SEQ = 64


def _load(model_ref: str):
    """Load an EuroBERT token-classification checkpoint (HF id or local dir)."""
    from transformers import AutoConfig, AutoModelForTokenClassification

    cfg = AutoConfig.from_pretrained(model_ref, trust_remote_code=True)
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            model_ref, trust_remote_code=True, dtype=torch.float32, attn_implementation="eager")
    except Exception:  # noqa: BLE001 — some custom_code models reject attn_implementation
        model = AutoModelForTokenClassification.from_pretrained(
            model_ref, trust_remote_code=True, dtype=torch.float32)
    return cfg, model.eval()


class LogitsWrapper(torch.nn.Module):
    """Expose the bare per-token logits (drop the HF output dataclass) for export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def _num_labels(cfg) -> int:
    n = getattr(cfg, "num_labels", None)
    if n:
        return int(n)
    labels = getattr(cfg, "id2label", None)
    return len(labels) if isinstance(labels, dict) else 0


def cmd_export(args) -> None:
    from coreai_torch import TorchConverter, get_decomp_table

    cfg, model = _load(args.model)
    wrapper = LogitsWrapper(model).eval()
    seq = int(args.seq)
    input_ids = torch.zeros(1, seq, dtype=torch.long)
    attention_mask = torch.ones(1, seq, dtype=torch.long)

    with torch.no_grad():
        ep = torch.export.export(wrapper, args=(input_ids, attention_mask), strict=False)
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(
        ep,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        entrypoint_name="main",
    )
    prog = conv.to_coreai()
    prog.optimize()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    aimodel = out / f"{out.name}.aimodel"
    prog.save_asset(aimodel)
    contract = {
        "entrypoint": "main",
        "seq_len": seq,
        "num_labels": _num_labels(cfg),
        "vocab_size": int(cfg.vocab_size),
        "hidden_size": int(cfg.hidden_size),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "model_type": cfg.model_type,
        "host_components": ["tokenizer"],
    }
    (out / "eurobert-contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"ok: lowered EuroBERT ({cfg.model_type}, {_num_labels(cfg)} labels) -> {aimodel}")
    print(f"ok: wrote {out / 'eurobert-contract.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="EuroBERT export (token-classification -> .aimodel)")
    ap.add_argument("phase", nargs="?", default="export", choices=["export"])
    ap.add_argument("--model", required=True, help="HF id or local snapshot dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    args = ap.parse_args()
    cmd_export(args)


if __name__ == "__main__":
    main()
