"""Technique smoke: complex RoPE -> real cos/sin rewrite (RFC F16 / playbook T1).

WHAT THIS EXERCISES: the numeric equivalence of the complex64 rotary application
`view_as_real(view_as_complex(x) * freqs_cis)` vs the real-arithmetic rewrite
`(x0*cos - x1*sin, x0*sin + x1*cos)` (with freqs_cis pre-split to real cos/sin). coreai_torch
has no complex dtype, so the export path MUST use the real form; this checks the two are the
same math. Central to the LingBot-Video DiT + Miril conversions this session.

PRECONDITION (playbook T1): also convert the freqs buffer itself to real (stack([cos, sin])),
or the graph re-crashes on the complex constant. Validate the rewrite BEFORE export (the
"fake parity" trap: never ship an unvalidated rewrite).

EXPLICIT EXCLUSION (RFC F12/F16): a green smoke is NOT proof coreai_torch lowers the real
graph on-device — it proves the real rewrite is numerically identical to the complex form.
Full-graph lowering + load is gated per composed bundle (protocol.loaded_on_ane).

REQUIRES: torch only (no Core AI toolchain). Skips cleanly if torch is absent."""
import sys

try:
    import torch
except ImportError as _exc:  # pragma: no cover
    print(f"SKIP: torch unavailable ({_exc})"); sys.exit(0)


def apply_complex(x, freqs_cis):          # freqs_cis: (S, D/2) complex
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))   # (B,S,H,D/2)
    f = freqs_cis[None, :, None, :]        # (1,S,1,D/2) -> broadcast over B,H
    return torch.view_as_real(x_c * f).flatten(3).type_as(x)


def apply_real(x, rot):  # rot = stack([cos, sin]) shape (2, S, D/2)
    cos = rot[0][None, :, None, :]; sin = rot[1][None, :, None, :]
    xr = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(3).type_as(x)


def main():
    torch.manual_seed(0)
    B, S, H, D = 1, 40, 12, 64            # (B, seq, heads, head_dim)
    x = torch.randn(B, S, H, D)
    freqs = torch.randn(S, D // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)      # complex64 (S, D/2)
    rot = torch.stack([freqs_cis.real, freqs_cis.imag], dim=0)  # real (2, S, D/2)
    a = apply_complex(x, freqs_cis)
    b = apply_real(x, rot)
    cos = torch.nn.functional.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()
    maxerr = (a - b).abs().max().item()
    print(f"complex-vs-real RoPE: cosine={cos:.10f} maxerr={maxerr:.2e}")
    ok = cos > 0.999999 and maxerr < 1e-4
    print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


main()
