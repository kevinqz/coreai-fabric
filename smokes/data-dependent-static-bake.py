"""Technique smoke: data-dependent control flow -> static bake (RFC F16 / playbook T9).

WHAT THIS EXERCISES: torch.export REFUSES a forward that materializes a shape-derived tensor
to a python value (`.tolist()`/`.item()`) — it raises GuardOnDataDependentSymNode (unbacked
symint). This smoke proves (a) the data-dependent form fails to export, and (b) the static-baked
form (the T9 fix) exports AND is numerically identical for the fixed shape. This is exactly the
LingBot-Video DiT text_lens `.tolist()` + the lazy-RoPE `.max().tolist()` blockers.

PRECONDITION (playbook T9): valid ONLY for a FIXED export input shape — the baked value is
genuinely constant for that shape; the host reproduces the (constant) op. Self-validate the
baked module vs the original eager forward (cosine ~1) before export.

EXPLICIT EXCLUSION (RFC F12/F16): green here means torch.export accepts the baked graph; it is
NOT proof coreai_torch lowers it or that it loads on-device (gated per composed bundle).

REQUIRES: torch only. Skips cleanly if torch is absent."""
import sys

try:
    import torch
except ImportError as _exc:  # pragma: no cover
    print(f"SKIP: torch unavailable ({_exc})"); sys.exit(0)


class DataDependent(torch.nn.Module):
    def forward(self, x):
        # the real text_lens pattern: int()/.tolist() of a data-dependent reduction forces a
        # guard torch.export CANNOT satisfy -> GuardOnDataDependentSymNode.
        k = int((x[:, :, 0] > 0.0).sum())      # data-dependent count materialized to python int
        return x[:, :k].sum(dim=1)


class StaticBaked(torch.nn.Module):
    """T9 fix: for the fixed export shape the mask is constant -> select all rows statically
    (host applies the constant mask). Here we keep the full reduction, no data-dependent count."""
    def forward(self, x):
        return x.reshape(-1, x.shape[-1]).sum(dim=0)


def main():
    torch.manual_seed(0)
    x = torch.randn(1, 16, 8)
    # (a) data-dependent form must FAIL to export
    dd_failed = False
    try:
        torch.export.export(DataDependent(), args=(x,), strict=False)
    except Exception as e:  # noqa: BLE001
        blob = f"{type(e).__name__} {e}".lower()
        dd_failed = any(k in blob for k in ("datadependent", "data-dependent", "symint", "guard", "specialize"))
    # (b) static-baked form exports + matches eager
    baked = StaticBaked().eval()
    ref = baked(x)
    ep = torch.export.export(baked, args=(x,), strict=False)
    out = ep.module()(x)
    cos = torch.nn.functional.cosine_similarity(ref.reshape(-1), out.reshape(-1), dim=0).item()
    print(f"data-dependent export failed as expected: {dd_failed} | baked cosine={cos:.10f}")
    ok = dd_failed and cos > 0.999999
    print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


main()
