# The `/reflect` ritual — the bounded, human-gated loop (RFC Phase 4, F4/F11)

This is the end-of-session step that turns the committed failure substrate
(`attempts/*.jsonl` from `coreai-fabric run`) + the measured-evaluation record
into bounded improvements to the playbook, the block vocabulary, and the error
signature table. It is **explicit** (a shipped ritual, not implicit discipline),
and **human-gated** at the commit. The loop **never blocks on a catalog PR
merge** (registration is already async by design).

There is **no autonomous acceptance**: a maintainer reviews and commits every
proposal. This is the refuted-finding guardrail — the loop's value/cost inverts
well above the solo-maintainer + agent-session scale the redteam measured.

## The loop

### 1. Mine

Read every `attempts/*.jsonl` written since the last reflect. Cluster by
`error_signature` (the distilled class from `coreai_fabric/error_signatures.py`):

- A cluster of `0x10004` on a new family → does T3 (graph-split) apply? Is the
  family's pass count bounded (the deployability precondition)?
- A cluster of `unclassified` → the signature table needs new entries. Each new
  class is a loop output (the `unclassified` fallback is itself a signal).
- A cluster of `parity_below_threshold` on a tier → does a playbook technique
  apply that wasn't used? (e.g. int8 experts dropping cosine → fp16 blocks.)

### 2. Propose (BOUNDED)

Each proposal is a diff limited to exactly these surfaces — **never** to
arbitrary driver internals:

- `docs/coreai-conversion-playbook.md` (a new `Tn` technique, or a precondition
  clarification on an existing one).
- `schema/blocks-vocab.yaml` (a new block id, with its one-line description).
- `coreai_fabric/error_signatures.py` (a new regex entry for an `unclassified`
  cluster).

Each proposal **names the attempt records that justify it** (the `ts` + recipe +
error_tail lines). A proposal without justifying records is rejected.

### 3. Validate

Run the `smokes/` proxy battery that keys to the touched technique. A proposal
that **regresses a smoke is rejected** — no commit. Remember the exclusion
contract (`smokes/README.md`): a green smoke is not a loadability guarantee; a
regression is still a real signal at the lowering/wiring level.

Also run `coreai-fabric validate` + `python -m pytest -q` + the generators'
`--check` (`generate_scorecard.py`, `generate_blocks_index.py`).

### 4. Human-reviewed commit

The diff lands only via a commit the maintainer reviews. The commit message cites
the F-codes and the attempt records (see the RFC traceability matrix in
`docs/superpowers/specs/2026-07-08-surgical-registry-design.md`).

The loop **does not** open or wait on a catalog PR here. Catalog registration is
batched and async by operating norm (playbook T7); the loop's proposals are
fabric-side.

## What this loop is NOT

- **Not autonomous.** No scheduled acceptance of registry changes without review.
- **Not cross-cell ranking.** The scorecard ranks within a protocol cell only (F2).
- **Not per-block cosine.** There is no per-block number to mine (F1/F6).
- **Not technique auto-apply.** `techniques.yaml` does not exist (F4); the
  playbook is the ACE context, the drivers are the executable form.
