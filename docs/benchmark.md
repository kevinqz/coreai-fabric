# Native benchmark (`bench` / `bench-submit`)

Fabric converts and verifies *fidelity* (Gate A/B). The **`bench` step** adds the
missing axis — real on-device **throughput** — by driving the catalog's SotA
benchmark protocol/runner against the freshly-converted bundle, so every
registered model can carry a **measured** tok/s number instead of a
README-scraped one.

## What it measures, and with what

- **Protocol:** the catalog's `benchmarks/protocol-config.json` v1.0 — a fixed
  128-token standard prompt, 256-token greedy decode, **3 warmup + 10 measured**
  runs, median. Metrics: `decode_throughput` (tok/s) and `time_to_first_token`
  (ms). Prefill cost is reported via TTFT, never folded into decode throughput.
- **Runner:** the catalog's `coreai-bench-runner`
  (`coreai-catalog/bench/CoreAIBenchRunner`), which drives Apple's
  `CoreAILanguageModels` engine on the **GPU**. This is a real on-device
  measurement — **not** fabric's Python Gate-B runtime (~0.16 tok/s, a wrong
  number for speed).
- **Scope:** LLM/VLM only (the runner implements the protocol's autoregressive
  metrics). Any other `bundle_kind` reports `not_run` honestly.

Fabric reuses the catalog's line assembler and validators
(`coreai_catalog.bench.assemble_benchmark_line` / `validate_runner_output`) — one
protocol, one assembler, no divergence.

## Requirements

- **macOS 27** with the **CoreAI framework** — build the runner from a catalog
  clone: `cd coreai-catalog/bench/CoreAIBenchRunner && swift build -c release`.
  On a machine where the stable Xcode SDK lacks the `CoreAI` module, use the
  macOS-27 SDK: `DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer`.
- A coreai-catalog clone (the same one `register` uses:
  `~/.cache/coreai-fabric/catalog`). Point `--catalog-path` elsewhere if needed.
- The bundle must exist (`coreai-fabric convert <id>` first).

## Usage

```sh
# 1. Measure (writes a durable `benchmark:` block into the recipe)
coreai-fabric bench qwen3-0.6b \
  --catalog-path ../coreai-catalog \
  --runner ../coreai-catalog/bench/CoreAIBenchRunner/.build/.../coreai-bench-runner

# 2. Inspect the assembled, schema-valid benchmarks.jsonl line
coreai-fabric bench-submit qwen3-0.6b --dry-run

# 3. Submit (signs + opens the benchmark-lane PR — see below)
coreai-fabric bench-submit qwen3-0.6b
```

`bench` writes the median decode/ttft + environment + self-check + device class
into `recipe.benchmark:` (durable — it survives a fresh clone, because
`build/` is gitignored). `publish` then fills the model card's
`Runtime throughput (tok/s)` line from it.

## Submission respects the catalog's separation

Benchmark numbers live only in the catalog's signed, append-only
`benchmarks.jsonl` — never inline-authoritative in a model entry. So `bench` and
`bench-submit` are **two lanes**:

- `bench-submit` assembles the line (`extraction_method: app_benchmark_protocol`,
  `verification_tier: unverified`), **signs it** (sigstore keyless), and opens a
  PR that touches **only `benchmarks.jsonl`** and adds **exactly one line** — the
  invariants the catalog's `benchmark-validate.yml` enforces.
- It refuses until the model's own PR is merged (the catalog's
  `model_id_exists` gate).

### Signing identity → auto-merge vs curator

Sigstore binds the signature to an OIDC identity:

- **GitHub Actions (ambient OIDC, `id-token: write`)** — the certificate carries
  your GitHub login; CI compares it to the PR author and **auto-merges** on a
  physics-clean, non-duplicate `signed_plausible` outcome. This is the intended
  production path:

  ```yaml
  permissions:
    id-token: write
    contents: write
  steps:
    - run: pip install "coreai-fabric[bench]" sigstore
    - run: coreai-fabric bench-submit ${{ inputs.recipe }} --catalog-path catalog
  ```

- **Local browser flow** — the certificate identity is your e-mail, which CI
  can't map to a PR author, so the submission lands in the **curator lane** (no
  auto-merge; a maintainer reviews).

Fabric never submits an unsigned `app_benchmark_protocol` row — the catalog's
relay lane rejects it by construction.
