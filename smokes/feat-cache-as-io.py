"""Technique smoke: stateful cache -> multi-tensor graph I/O (RFC F16 / playbook T10).

WHAT THIS EXERCISES: a streaming decoder threads a stateful cache across chunks (WAN VAE
causal-conv feat_cache, or a KV-cache) — the export ships the SUBSEQUENT-chunk graph
`(chunk, *cache_in) -> (out, *cache_out)` with the cache as N positional tensor I/O. This proves
that reformulating an in-place stateful step as explicit cache-in/cache-out (a) is numerically
identical to the stateful reference, and (b) exports (pytree tuple I/O). Mirrors the
LingBot-Video streaming VAE (32 cache tensors threaded).

PRECONDITION (playbook T10): reach STEADY STATE before sampling the cache template (prime
first-chunk + one subsequent chunk); thread only TENSOR slots, bake non-tensor sentinels;
detach() cache tensors before .numpy() in the parity loop.

EXPLICIT EXCLUSION (RFC F12/F16): green means the reformulation matches + exports; NOT that a
32-tensor-I/O graph lowers/loads on-device (gated per composed bundle).

REQUIRES: torch only. Skips cleanly if torch is absent."""
import sys

try:
    import torch
except ImportError as _exc:  # pragma: no cover
    print(f"SKIP: torch unavailable ({_exc})"); sys.exit(0)


class StatefulConv(torch.nn.Module):
    """A causal temporal conv that keeps the last frame as internal state (the pattern T10 unrolls)."""
    def __init__(self, c):
        super().__init__(); self.w = torch.nn.Conv1d(c, c, 2, bias=False); self.state = None
    def step(self, x):  # x: (B, C, 1)
        prev = self.state if self.state is not None else x
        y = self.w(torch.cat([prev, x], dim=-1))
        self.state = x
        return y


class CacheIO(torch.nn.Module):
    """The T10 reformulation: cache is explicit I/O, no internal state."""
    def __init__(self, conv): super().__init__(); self.w = conv.w
    def forward(self, x, cache_in):        # cache_in = previous frame (B,C,1)
        y = self.w(torch.cat([cache_in, x], dim=-1))
        return y, x                         # (out, cache_out)


def main():
    torch.manual_seed(0)
    B, C = 1, 4
    sc = StatefulConv(C).eval()
    x0 = torch.randn(B, C, 1); x1 = torch.randn(B, C, 1)
    # reference: stateful stream (chunk0 primes, chunk1 is the subsequent chunk)
    with torch.no_grad():
        sc.step(x0)                        # prime -> state = x0
        ref = sc.step(x1)                  # subsequent chunk
    # T10: explicit cache I/O, cache_in = x0 (the primed steady-state cache)
    cio = CacheIO(sc).eval()
    with torch.no_grad():
        out, cache_out = cio(x1, x0)
    cos = torch.nn.functional.cosine_similarity(ref.reshape(-1), out.reshape(-1), dim=0).item()
    # and it exports with tuple I/O
    ep = torch.export.export(cio, args=(x1, x0), strict=False)
    exported = len(list(ep.graph.nodes)) > 0
    cache_ok = torch.allclose(cache_out, x1)
    print(f"cache-IO vs stateful: cosine={cos:.10f} | cache_out correct={cache_ok} | exports={exported}")
    ok = cos > 0.999999 and cache_ok and exported
    print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


main()
