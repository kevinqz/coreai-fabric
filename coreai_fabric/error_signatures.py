"""RFC Phase 1 (F8): distill a conversion's stderr/stdout into a stable
``error_signature`` class so the failure substrate (attempts/*.jsonl) can be
clustered by weakness.

Ordered regex table (first match wins). ``unclassified`` is the fallback and is
itself a loop input — it means the table needs a new entry, surfaced by the
/reflect ritual. The signatures are deliberately coarse: they name a *failure
class* (the ANE program ceiling, a dtype the toolchain can't lower, an OOM), not
a stack frame, because that is the granularity at which a technique applies.

Each signature maps to an ``outcome`` the run loop records:
- ``blocked``   — an external ceiling the recipe cannot code around in-fabric
                  (ANE 0x10004 program ceiling, license, toolchain-skew).
- ``failed``    — attempted, did not pass; a code/config change can plausibly fix it.
- ``parity_below_threshold`` — the export RAN and produced a number under the bar.
- ``unclassified`` — recorded as-is; flagged as a loop input to extend the table.
"""
from __future__ import annotations

import re

#: (signature, outcome, [regex, ...]) — first matching regex wins. Ordered so the
#: most specific signatures precede the generic ones.
TABLE: list[tuple[str, str, list[re.Pattern[str]]]] = [
    ("0x10004", "blocked", [
        re.compile(r"0x10004", re.IGNORECASE),
        re.compile(r"appleneuralengine\s+program\s+load\s+failure", re.IGNORECASE),
        re.compile(r"program\s+load\s+failure", re.IGNORECASE),
        re.compile(r"\bane\b.*\b(ceiling|limit|exceed|overflow)\b", re.IGNORECASE),
    ]),
    ("complex128", "failed", [
        re.compile(r"complex128", re.IGNORECASE),
        re.compile(r"unsupported.*complex", re.IGNORECASE),
    ]),
    ("FoldMultiplyIntoSDPAScale", "failed", [
        re.compile(r"FoldMultiplyIntoSDPAScale", re.IGNORECASE),
        re.compile(r"scaled_dot_product_attention.*scale", re.IGNORECASE),
    ]),
    ("versioned_IR", "blocked", [
        re.compile(r"versioned\s*IR", re.IGNORECASE),
        re.compile(r"unsupported\s*IR\s*version", re.IGNORECASE),
        re.compile(r"coreai-torch.*version", re.IGNORECASE),
        re.compile(r"incompatible.*toolchain", re.IGNORECASE),
    ]),
    ("OOM", "blocked", [
        re.compile(r"\boom\b|out\s+of\s+memory", re.IGNORECASE),
        re.compile(r"jetsam", re.IGNORECASE),
        re.compile(r"mpsgraph.*(overflow|exceed)", re.IGNORECASE),
        re.compile(r"cannot\s+allocate.*memory", re.IGNORECASE),
    ]),
    ("import_error", "failed", [
        re.compile(r"ModuleNotFoundError|ImportError", re.IGNORECASE),
        re.compile(r"No module named", re.IGNORECASE),
    ]),
    ("accelerator_init", "blocked", [
        re.compile(r"(cuda|mps|gpu).*(init|not available|not found)", re.IGNORECASE),
        re.compile(r"no\s+metal\s+device", re.IGNORECASE),
    ]),
    ("parity_below_threshold", "parity_below_threshold", [
        re.compile(r"parity.*(below|failed|under)", re.IGNORECASE),
        re.compile(r"gate\s*b.*(failed|below)", re.IGNORECASE),
        re.compile(r"cosine.*below.*threshold", re.IGNORECASE),
    ]),
    ("vocab_tokenizer_mismatch", "failed", [
        re.compile(r"(vocab|tokenizer).*(mismatch|size|length)", re.IGNORECASE),
        re.compile(r"token.*(out of range|index error)", re.IGNORECASE),
    ]),
    ("data_dependent", "failed", [
        # torch.export can't specialize a data-dependent value (unbacked symint u0):
        # .tolist()/.item() on shape-derived scalars, boolean masked_select, lazy
        # table rebuilds. Fix = playbook T9 (static bake / graph-cut-before).
        re.compile(r"GuardOnDataDependentSymNode", re.IGNORECASE),
        re.compile(r"data-dependent\s+(expression|value|symnode)", re.IGNORECASE),
        re.compile(r"could\s+not\s+(guard\s+on|extract\s+specialized).*\bu\d", re.IGNORECASE),
        re.compile(r"unbacked\s+symint", re.IGNORECASE),
    ]),
    ("license_blocked", "blocked", [
        re.compile(r"gated\s+repo|access.*denied.*license", re.IGNORECASE),
        re.compile(r"must.*accept.*terms", re.IGNORECASE),
    ]),
]

UNCLASSIFIED = "unclassified"


def classify(error_text: str) -> str:
    """Return the distilled error-signature class for a chunk of error text.

    First-match-wins over TABLE. ``unclassified`` when nothing matches (a signal
    the table needs a new entry — a /reflect loop input). Never raises."""
    if not error_text:
        return UNCLASSIFIED
    for sig, _outcome, patterns in TABLE:
        if any(p.search(error_text) for p in patterns):
            return sig
    return UNCLASSIFIED


def outcome_for(signature: str) -> str:
    """The run-loop ``outcome`` for a signature (blocked/failed/parity_below_threshold).

    ``unclassified`` (and any unknown signature) maps to ``failed`` — an unknown
    failure is treated as a fixable attempt by default, not a silent block."""
    for sig, outcome, _patterns in TABLE:
        if sig == signature:
            return outcome
    return "failed"


def outcome(exit_code: int, error_text: str) -> tuple[str, str]:
    """Combine a process exit code + stderr into ``(outcome, signature)``.

    Exit 0 -> ('converted', 'ok'). Non-zero -> classify the stderr; the signature
    determines whether the failure is ``blocked`` (external ceiling) or ``failed``."""
    if exit_code == 0:
        return ("converted", "ok")
    sig = classify(error_text)
    return (outcome_for(sig), sig)
