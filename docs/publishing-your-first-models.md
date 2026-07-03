# Publishing your first models with coreai-fabric

A concrete, honest handoff for converting + publishing LLMs to **your own**
Hugging Face namespace (`kevinqz`). It marks clearly what is already prepared
for you vs. the few steps only you can run (they need your Hugging Face
credentials or your machine's toolchain).

## What's already done for you

- **3 production recipes are scaffolded, validated, and committed** — a small
  Apache-2.0 size ladder, all using Apple's tested 4bit macOS presets:

  | recipe id | upstream | preset ctx | publishes to |
  |---|---|---|---|
  | `qwen3-0.6b` | Qwen/Qwen3-0.6B | 8192 | `kevinqz/Qwen3-0.6B-CoreAI` |
  | `qwen2.5-1.5b-instruct` | Qwen/Qwen2.5-1.5B-Instruct | 32768 | `kevinqz/Qwen2.5-1.5B-Instruct-CoreAI` |
  | `qwen3-4b` | Qwen/Qwen3-4B | 40960 | `kevinqz/Qwen3-4B-CoreAI` |

  All three pass `coreai-fabric validate`, and their generated catalog entries
  pass the **live catalog's** validate + audit + io_contract gates
  (cross-contract CI). `qwen3-0.6b` was additionally run end-to-end on real
  hardware — see `docs/validation-log.md`.

  > **Namespace note.** All three publish to **your** namespace (`kevinqz`).
  > Do NOT use `coreai-community` — that is a real, separate HF org (members
  > include Hugging Face's `pcuenq`) that mirrors community CoreAI conversions;
  > you are not a member, so a publish there would fail. fabric's `--namespace`
  > default of `coreai-community` is a footgun for that reason — always pass
  > your own `--namespace`.

- **Want different models?** Any short-name from Apple's registry works. List
  them with `coreai.model.registry --list-models --type llm`. The permissive
  (Apache-2.0) macOS presets are: `qwen3-0.6b`, `qwen2.5-1.5b-instruct`,
  `qwen3-4b`, `qwen3-8b`, `qwen3-coder-30b-a3b-instruct`,
  `mistral-7b-instruct-v0.3`, `mixtral-8x7b-instruct-v0.1`, `gpt-oss-20b`.
  (`gemma3-4b-it` / `gemma3-12b-it` are `review_required` — the Gemma license
  needs `--acknowledge-license-review` on publish.) Scaffold one with:
  ```bash
  coreai-fabric new <owner/name> --namespace kevinqz \
      --tool coreai.llm.export --apple-registry-name <short-name>
  ```

## What only you can do

These need **your** credentials or your machine and cannot be done for you:
install the Apple toolchain + GitHub CLI, connect Hugging Face, and run the
upload/PR steps.

### Step 1 — Install the toolchain (one time, your Mac)

Apple Silicon + macOS required. The production exporter is NOT on PyPI.

```bash
# Apple's production CLIs (coreai.llm.export, coreai.model.registry, …)
git clone https://github.com/apple/coreai-models
python -m venv .venv && source .venv/bin/activate
pip install ./coreai-models/python

# fabric with the convert + publish extras (after PR #1 merges; or `pip install -e .`
# from your local checkout of this repo on the feature branch)
pip install "coreai-fabric[convert,hf]"

# GitHub CLI — the `register` step (Step 4) opens the catalog PR via `gh`.
# It is a system binary, NOT a pip package. If you don't already have it:
brew install gh && gh auth login      # needs `repo` scope to open/fork the catalog PR
```

### Step 2 — Connect Hugging Face (this is the "how do I connect HF?" step)

fabric uploads to **your** namespace using a token you create. You need a
**Write** token (it has to create the repo and upload files):

1. Open <https://huggingface.co/settings/tokens>.
2. **Create new token** → Type **Write** (or a fine-grained token with
   *Write access to contents/settings of your repos*). Copy it (`hf_…`).
3. Log the token into your local CLI:
   ```bash
   hf auth login          # paste the token when prompted; answer "Y" to git credential
   ```
   (Non-interactive alternative: `export HF_TOKEN=hf_xxx` in your shell.)
4. Verify the connection:
   ```bash
   hf auth whoami         # must print: kevinqz
   ```

That's the whole Hugging Face connection. fabric reads this token
automatically — you never pass it on the command line. **fabric never hosts
weights**; it only uploads to the `--namespace` you set (here, `kevinqz`).

### Step 3 — Run the pipeline (per model)

Each recipe walks convert → verify → publish → register. Example, `qwen3-0.6b`:

```bash
coreai-fabric convert  qwen3-0.6b          # runs coreai.llm.export → 4bit KV-cache asset (~320 MB)
coreai-fabric verify   qwen3-0.6b          # Gate A passes; Gate B = not_run (expected — see below)
coreai-fabric publish  qwen3-0.6b --allow-unverified-parity   # uploads to huggingface.co/kevinqz/…
coreai-fabric register qwen3-0.6b --catalog-path ../coreai-catalog   # opens the catalog PR (needs gh)
```

Repeat for `qwen2.5-1.5b-instruct` and `qwen3-4b`. `register` needs a local
clone of `coreai-catalog` (`--catalog-path`) **and the GitHub CLI authenticated**
(`gh auth login`, Step 1); it validates the entry, replays the catalog's own CI
locally, and opens a PR (forking if you lack push access). To preview without
`gh` first, run `coreai-fabric register <id> --dry-run` (prints the YAML, opens
nothing). After the PR merges: `coreai-fabric register <id> --mark-merged`.

## The one honest caveat: `--allow-unverified-parity`

Every production publish currently needs `--allow-unverified-parity`, and this
is **correct, not a workaround**:

- The production asset is quantized (4bit) + stateful (KV-cache). Its correct
  Gate B is **benchmark accuracy** vs upstream, not raw logit fidelity.
- Apple's evaluator for that (`coreai.llm.eval`) is a **stub** in
  coreai-models 0.1.0 ("Evaluation support is coming soon"), so Gate B honestly
  reports `not_run` — fabric refuses to fake a parity number.
- The model card records this plainly, and the catalog entry lands as
  `status: needs_review`. When Apple ships the evaluator, drop the flag and
  Gate B runs for real.

So: Gate A (structure) is proven; numeric quality parity waits on upstream. You
are publishing a structurally-verified, honestly-labelled artifact.

## SotA distribution: own your namespace, mirror into `coreai-community`

`coreai-community` (huggingface.co/coreai-community) is the community org for
Apple CoreAI assets — members include Hugging Face's `pcuenq`. You are not a
member yet, but the org is open ("**Join this org**" on its page). The SotA
move is NOT to publish straight into it — it is the pattern the org already
uses: **its repos are mirrors of individual authors' namespaces** (their
commit history literally reads "Mirror of `<author>/<model>`").

So the durable, attribution-preserving flow is:

1. **Publish to your own namespace** (`kevinqz/<model>-coreai`) — you own the
   canonical repo, keep attribution, and can iterate. This is Steps 1–4 above.
2. **Request to join** `coreai-community` (the "Join this org" button), or ask a
   maintainer to mirror your repo — exactly how the existing entries got there.
3. **Mirror** your canonical repo into the org once you're in (or let a
   maintainer). The org entry is a discovery surface; your namespace stays the
   source of truth.
4. **Index both in the catalog.** `coreai-catalog` points at your namespace as
   provenance and can note the community mirror — the catalog is the neutral
   index that ties the source repo and the community copy together.

Why this beats publishing directly into the org: your name stays on the work,
your repo survives any org change, and the community org becomes a distribution
layer rather than a single point of ownership. If you'd rather have a branded
home, create your own org (e.g. `coreai-br`) and mirror there instead — the
same own-source-of-truth principle applies.

### The organization scheme fabric now enforces

So a `kevinqz` repo and its `coreai-community` mirror are one clean, discoverable,
honest unit, fabric standardizes:

- **Repo name:** `<UpstreamModelName>-CoreAI` — matches the community's own
  convention (`Qwen3-0.6B-CoreAI`, `MiniCPM-V-4.6-CoreAI`), so your repo and its
  mirror share one name. (Was lowercase `<id>-coreai`.)
- **`base_model` + `base_model_relation: quantized`** in the card frontmatter.
  This is the key discoverability lever: HF then lists your conversion **on the
  upstream model's page** as a quantized derivative. Note we use `quantized` —
  the *correct* relation; the community's own cards show `finetune`, which is
  wrong for a quantized export. When the preset is uncompressed (`none`), fabric
  omits the relation rather than mislabel it.
- **Consistent tags:** `coreai`, `core-ai`, `coreai-fabric`, `aimodel`, `apple`,
  `apple-silicon`, `on-device`, the `bundle_kind` (`llm`/`asr`/`vlm`/…), and the
  quantization (`4bit`) — findable by exactly what it is. `library_name: coreai`.
- **The catalog is the neutral index** that ties the three identities together:
  upstream (provenance) → your namespace (source of truth) → community mirror
  (distribution). One entry, three links, no ambiguity about who made what.

Two things fabric does NOT do for you (HF-account actions): **HF Collections**
— group your CoreAI repos into a Collection per capability ("CoreAI LLMs",
"CoreAI ASR") for a clean landing page — and the **mirror push** itself. Both
are one-time clicks/API calls under your account.
