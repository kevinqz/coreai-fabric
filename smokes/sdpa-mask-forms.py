"""Technique smoke: SDPA mask forms (RFC F16 / playbook T2).

WHAT THIS EXERCISES: the numeric equivalence of the four attention forms the
FoldMultiplyIntoSDPAScale workaround chooses between — mask-free SDPA, a
constant additive-zero mask, a data-dependent all-true BOOL keep-mask, and eager
`softmax(QKᵀ/√d)·V`. All four must produce the SAME output; the fix swaps the
GRAPH OP (a foldable additive scale-multiply → a non-foldable where/select)
WITHOUT changing the math. This is the check the /reflect loop runs after
touching the SDPA-mask technique (T2).

PRECONDITION (playbook T2): the bool keep-mask variant is the fallback for models
with NO eager attention path; where an eager path exists, prefer it (no SDPA →
the fold can't trigger). This smoke validates that the bool keep-mask is a
numeric no-op vs mask-free SDPA, i.e. safe to substitute.

EXPLICIT EXCLUSION — a green smoke is NOT proof the on-device fold is avoided
(RFC F12/F16): the MPSGraph FoldMultiplyIntoSDPAScale SEGFAULT only manifests in
the Core AI compiler on macOS 27, which this pure-torch check cannot see. It
proves the mask forms are MATHEMATICALLY equivalent, not that the lowered graph
loads. A constant additive-zero mask is included precisely because it folds away
on-device and re-crashes — here it only confirms the math matches.

REQUIRES: torch only (no Core AI toolchain). Skips with a clear message if torch
is absent."""
import sys

try:
    import torch
    import torch.nn.functional as F
except ImportError as _exc:  # pragma: no cover
    print(f"SKIP: sdpa-mask-forms smoke needs torch (import failed: {_exc.name}).")
    sys.exit(0)

torch.manual_seed(0)
B, H, Lq, Lk, D = 2, 4, 7, 7, 16
q = torch.randn(B, H, Lq, D, dtype=torch.float32)
k = torch.randn(B, H, Lk, D, dtype=torch.float32)
v = torch.randn(B, H, Lk, D, dtype=torch.float32)


def eager(q, k, v, mask=None):
    scores = (q @ k.transpose(-1, -2)) / (D ** 0.5)
    if mask is not None:
        scores = scores + mask
    return F.softmax(scores, dim=-1) @ v


# (a) mask-free SDPA — the form that segfaults on macOS 27 at load.
out_sdpa_free = F.scaled_dot_product_attention(q, k, v, attn_mask=None)

# (b) constant additive-zero mask — folds away on-device (re-crashes); math is a no-op.
zero_add = torch.zeros(Lq, Lk, dtype=torch.float32)
out_zero_add = F.scaled_dot_product_attention(q, k, v, attn_mask=zero_add)

# (c) data-dependent all-true BOOL keep-mask — the fix: a where/select, not a
#     foldable scale-multiply. Built so it can't be constant-folded away.
flag = (k.abs().sum() >= -1.0)  # always True, but data-dependent
keep = flag.reshape(1, 1).expand(Lq, Lk)  # bool keep-mask
out_bool_keep = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)

# (d) eager manual attention — the cleanest fix when the model has the path.
out_eager = eager(q, k, v)

ref = out_sdpa_free
worst = 0.0
for name, out in (("zero-add", out_zero_add), ("bool-keep", out_bool_keep), ("eager", out_eager)):
    d = (out - ref).abs().max().item()
    worst = max(worst, d)
    print(f"  {name:10} max|Δ| vs mask-free SDPA = {d:.2e}")

TOL = 1e-5
if worst <= TOL:
    print(f"PASS: all SDPA mask forms numerically equivalent (worst Δ {worst:.2e} <= {TOL:.0e}).")
    sys.exit(0)
print(f"FAIL: an SDPA mask form diverged (worst Δ {worst:.2e} > {TOL:.0e}) — "
      "the bool keep-mask substitution is NOT a numeric no-op here.", file=sys.stderr)
sys.exit(1)
