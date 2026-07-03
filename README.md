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
  parity is `not_run` until a real runner measures it — never faked. Where the
  Apple converter's CLI interface could not be verified offline, the single
  assumption is isolated and TODO-marked (`coreai_fabric/convert.py`).

## The pipeline

| Command | Does | Requires |
|---|---|---|
| `coreai-fabric new <hf_repo>` | Scaffold `recipes/<id>.yaml` from HF API metadata | network (or `--offline` + flags) |
| `coreai-fabric validate [id]` | Schema + license triage, aggregated errors | nothing |
| `coreai-fabric convert <id>` | Upstream → `.aimodel` via the Apple toolchain adapter; writes a conversion manifest | **macOS + apple/coreai-torch** |
| `coreai-fabric verify <id>` | Gate A: bundle structure + metadata sanity. Gate B: numeric parity vs upstream (cosine thresholds from the recipe); writes `parity-report.json` | Gate B: **macOS + Core AI runtime + a parity runner** |
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
coreai-fabric new Qwen/Qwen3-0.6B  # scaffold your own
```

## Seed recipes

Three draft recipes covering diverse modalities, grounded in live-verified
upstream metadata (they have **not** been converted yet — that requires the
Apple toolchain on macOS):

| Recipe | Upstream | Modality | License |
|---|---|---|---|
| `qwen3-0.6b` | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | text → text (LLM) | apache-2.0 |
| `whisper-large-v3-turbo` | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | audio → transcript | mit |
| `da3-small` | [depth-anything/DA3-SMALL](https://huggingface.co/depth-anything/DA3-SMALL) | image → depth map | apache-2.0 |

## Relationship to the ecosystem

- **[coreai-catalog](https://github.com/kevinqz/coreai-catalog)** — the
  registry. Fabric is one of its upstream sources (`source_group: fabric`);
  `register` opens catalog PRs with pinned, digest-verified provenance.
- **john-rocky/coreai-model-zoo** — the original community conversion zoo.
  The catalog continues to index it as a reference upstream; fabric is
  independent of it and takes nothing from it.
- **Hugging Face** — hosts all published bytes, always under the publisher's
  own namespace (default org: `coreai-community`).
- **Apple (apple/coreai-models, apple/coreai-torch)** — provides the
  toolchain and runtime that `convert` drives. Fabric is not affiliated with
  or endorsed by Apple.

## Verification gates

- **Gate A — structural** (runs anywhere): expected bundle files present,
  `metadata.json` parses, metadata agrees with recipe expectations.
- **Gate B — numeric parity** (macOS + Core AI runtime): cosine similarity vs
  the upstream model per `docs/parity-protocol.md`; thresholds live in each
  recipe (convention: ≥ 0.999, plus greedy-token-exact for LLMs). The Swift
  runner is not implemented yet; Gate B reports `not_run` honestly until it is.

## Contributing & governance

Contributions are **recipes and parity reports** — see
[CONTRIBUTING.md](CONTRIBUTING.md). Merge rules are checkable, not vibes —
see [GOVERNANCE.md](GOVERNANCE.md). MIT licensed.
