# smokes/ — the loop's cheap local evaluator (RFC Phase 1d, F16)

These are the **technique-keyed proxy battery** the `/reflect` loop runs after
touching a playbook technique or an `error_signatures.py` entry. They are NOT
fleet re-verification — they are the cheap random-weight checks that exercise one
technique's lowering/wiring at tiny scale.

## The exclusion contract (F12/F16)

**Every smoke header states explicitly what it CANNOT see.** A green smoke is
never a loadability or parity guarantee. The exclusions are scale-dependent
effects that only bite at full depth:

- **ANE 0x10004 program ceiling** — a 2- or 4-layer graph loads where a 36-layer
  one does not. The smoke cannot see this.
- **fp16 load ceiling** — the same MoE stack loads at L=12 fp16, fails at L=18.
- **per-checkpoint static-shape identity** — three pi05 fine-tunes are three
  different static-shape identities; a smoke on a random config does not pin one.

Loadability is gated per **composed bundle** (`protocol.loaded_on_ane`), never
inferred from a smoke green or a block-level fact.

## Running

The smokes self-`SKIP` (exit 0 with a message) when the convert toolchain
(`coreai_torch` + `torch` + the model source under `models/`) is absent — so CI,
which has no Apple toolchain, passes cleanly, and a maintainer with the toolchain
runs them locally:

```bash
python smokes/moe-dense-fusion-lowering.py     # playbook T2: MoE dense-fusion lowering
python smokes/graph-split-chaining.py          # playbook T3: graph-split host-chain wiring
```

## Adding a smoke

Promote a scratchpad random-weight check here when it keys to a technique, and
write its exclusion header in the same commit. A smoke that does not state what
it cannot see is incomplete by contract.
