# coreai-fabric

**The conversion fabric for Apple Core AI — recipes in, provenance-verified
`.aimodel` out, indexed by [coreai-catalog](https://github.com/kevinqz/coreai-catalog).**

Apple ships recipes, not artifacts: `apple/coreai-models` provides export
recipes and a Swift runtime, `apple/coreai-torch` the converter — and Apple
publishes zero `.aimodel` files. Artifact production is delegated to the
community. coreai-fabric is the first-party, agent-first pipeline for that
layer: a strict recipe contract, an executable convert→verify→publish→register
loop, and verifiable provenance at every step.

```
                         recipes in                     provenance-verified .aimodel out
 upstream HF repos ────────────────▶  coreai-fabric  ────────────────▶  publisher's OWN HF repo
 (Qwen, openai, ...)                  convert · verify · publish              (hosts the bytes)
                                            │                                       │
                                            │ register (PR with pinned                │
                                            │ revision + per-file sha256)             │
                                            ▼                                       │
                                     coreai-catalog  ◀────────── indexes ───────────┘
                                     (index, never a host)
                                            │
                                            │ also indexes (reference upstream)
                                            ▼
                              john-rocky/coreai-model-zoo
                        (independent community zoo — indexed,
                         not a dependency of this pipeline)
```

## Principles

- **Index-not-host, end to end.** Fabric never hosts weights. `build/` is
  gitignored; artifacts are uploaded to each publisher's own Hugging Face
  namespace, and only metadata + digests flow into the catalog.
- **Agent-first.** Every step is a CLI command with aggregated, actionable
  errors. The full loop — including opening the catalog PR — runs without
  out-of-band human knowledge. Start at [AGENTS.md](AGENTS.md).
- **Provenance is data, not fiction.** Catalog entries generated here carry
  `source_group: fabric`, a pinned HF `revision`, per-file `sha256` digests,
  and `provenance.converted_by {tool, version, recipe_url}` — no fabricated
  GitHub coordinates, no unverifiable claims.
- **Never fabricate.** Unknowable fields stay absent or `unknown`. Gate B
  parity is `not_run` until a real runner measures it — never faked. The
  converter interface is no longer an assumption: it was verified on real
  hardware (macOS 26, Apple Silicon, 2026-07-03 — see `docs/toolchain-notes.md`
  and `docs/validation-log.md`).

## The pipeline

| Command | Does | Requires |
|---|---|---|
| `coreai-fabric new <hf_repo>` | Scaffold `recipes/<id>.yaml` from HF API metadata | network (or `--offline` + flags) |
| `coreai-fabric validate [id]` | Schema + license triage, aggregated errors | nothing |
| `coreai-fabric convert <id>` | Upstream → `.aimodel` via the Apple toolchain adapter; writes a conversion manifest | **macOS/arm64 + `pip install ".[convert]"`** (`coreai-torch` on PyPI is a library, not a CLI — fabric ships the `coreai-fabric-llm-export` executable over it; verified on macOS 26) |
| `coreai-fabric verify <id>` | Gate A: bundle structure + metadata sanity. Gate B: numeric parity vs upstream (cosine thresholds from the recipe); writes `parity-report.json` | Gate B: **macOS + a parity runner** (fabric ships `coreai-fabric-parity-runner`; the runtime in the coreai-core wheel executes assets on macOS 26) |
| `coreai-fabric publish <id>` | Upload bundle + normalized model card + reports to the publisher's own HF namespace; pins the resulting revision into the recipe | `pip install ".[hf]"`, `hf auth login` |
| `coreai-fabric register <id>` | Generate + schema-validate coreai-catalog entries, open PR `fabric/add-<id>` | catalog clone, `gh auth login` |
| `coreai-fabric list` / `status` | Recipe inventory with pipeline stage | nothing |

## Quick start

```bash
git clone https://github.com/kevinqz/coreai-fabric
cd coreai-fabric
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"

coreai-fabric list                 # seed recipes and their stages
coreai-fabric validate             # all green
coreai-fabric new Qwen/Qwen3-0.6B --namespace <your-hf-user>  # scaffold your own
```

> `--namespace` defaults to your logged-in HF user (`hf whoami`). Fabric refuses
> to scaffold into a shared org (e.g. `coreai-community`) without `--i-am-mirroring`
> — your own namespace is the source of truth; a shared org is a mirror.

## Recipes

Recipes span the fabric lanes, grounded in live-verified upstream metadata and converted +
parity-checked on Apple Silicon (see `docs/validation-log.md` for the per-model conversion + parity
lineage). Many are published to the publisher's own HF namespace (`kevinqz/*-CoreAI`) and indexed in
[coreai-catalog](https://github.com/kevinqz/coreai-catalog); the rest are drafts or blocked on a
lane/arch gap. Artifacts are not committed (`build/` is disposable) — each published repo carries the
`.aimodel`, model card, `parity-report.json`, and (for Gemma derivatives) the mandated `NOTICE`.

Lanes:

- **LLM** — `coreai.llm.export` (arch-gated: gemma3_text, gpt_oss, mistral, mixtral, qwen2, qwen3,
  qwen3_moe, qwen3_vl). Gate B = greedy-token parity.
- **VLA / action** — ACT · Diffusion · VQ-BeT · pi0 · pi05 · SmolVLA · **pi0fast** (autoregressive,
  StaticCache decode). Gate B = action / greedy-token parity.
- **VLM** — `coreai.vlm.export` (qwen3_vl family). **diffusion** — `coreai.diffusion.export`.

Run `coreai-fabric list` for the live inventory + per-recipe pipeline stage, and `coreai-fabric
status` for what's next.

## Relationship to the ecosystem

- **[coreai-catalog](https://github.com/kevinqz/coreai-catalog)** — the
  registry. Fabric is one of its upstream sources (`source_group: fabric`);
  `register` opens catalog PRs with pinned, digest-verified provenance.
- **john-rocky/coreai-model-zoo** — the original community conversion zoo.
  The catalog continues to index it as a reference upstream; fabric is
  independent of it and takes nothing from it.
- **Hugging Face** — hosts all published bytes, always under the publisher's
  own namespace (defaults to your logged-in HF user; never a shared org).
- **Apple (apple/coreai-models, apple/coreai-torch)** — provides the
  toolchain and runtime that `convert` drives. Fabric is not affiliated with
  or endorsed by Apple.

## Verification gates

- **Gate A — structural** (runs anywhere): expected bundle files present,
  `metadata.json` parses, metadata agrees with recipe expectations.
- **Gate B — numeric parity** (macOS + Core AI runtime): cosine similarity vs
  the upstream model per `docs/parity-protocol.md`; thresholds live in each
  recipe (convention: ≥ 0.999, plus greedy-token-exact for LLMs). Fabric ships
  `coreai-fabric-parity-runner` (per_token_logit_cosine, validated on real
  hardware); without a configured runner Gate B reports `not_run` honestly.

## Contributing & governance

Contributions are **recipes and parity reports** — see
[CONTRIBUTING.md](CONTRIBUTING.md). Merge rules are checkable, not vibes —
see [GOVERNANCE.md](GOVERNANCE.md). MIT licensed.
