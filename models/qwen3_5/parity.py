"""Two-venv greedy parity for the Qwen3.5 lane: Gate A token-exact, B first-token
logit cosine, C vision-feature cosine, D recurrent-state cosine. Long context uses
a margin rule + rollout sanity, not bit-exact >=0.999 (fp16 noise compounds).
M-RoPE is kept (unlike models/vlm/parity.py, which strips it)."""
from __future__ import annotations
import torch


def state_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    fa, fb = a.flatten().float(), b.flatten().float()
    return float(torch.nn.functional.cosine_similarity(fa, fb, dim=0))


def token_match_rate(ref: list[int], got: list[int]) -> float:
    n = min(len(ref), len(got))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if ref[i] == got[i]) / n
