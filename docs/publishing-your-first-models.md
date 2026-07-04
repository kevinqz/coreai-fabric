# Publishing your first models with coreai-fabric

A concrete, honest handoff for converting + publishing LLMs to **your own**
Hugging Face namespace (`kevinqz`). It marks clearly what is already prepared
for you vs. the few steps only you can run (they need your Hugging Face
credentials or your machine's toolchain).

## What's already done for you

- **One conversion is already published + live**, and an Apache-2.0 size ladder
  is scaffolded, validated, and committed. Each Qwen model has an **int8
  high-fidelity tier** (lead with this) and an **int4 size tier**:

  | recipe id | upstream | tier | ctx | publishes to | status |
  |---|---|---|---|---|---|
  | `qwen3-0.6b-int8` | Qwen/Qwen3-0.6B | int8 (high-fidelity) | 8192 | `kevinqz/Qwen3-0.6B-CoreAI` | **published + verified** |
  | `qwen3-0.6b` | Qwen/Qwen3-0.6B | int4 (size) | 8192 | same repo, `int4/` | draft (measured-lossy, 79%) |
  | `qwen3-4b-int8` · `qwen3-4b` | Qwen/Qwen3-4B | int8 · int4 | 40960 | `kevinqz/Qwen3-4B-CoreAI` | draft (int8 **unmeasured**) |
  | `qwen2.5-1.5b-instruct-int8` · `…` | Qwen/Qwen2.5-1.5B-Instruct | int8 · int4 | 32768 | `kevinqz/Qwen2.5-1.5B-Instruct-CoreAI` | draft (int8 **unmeasured**) |

  `qwen3-0.6b-int8` was run end-to-end on real hardware, **passed Gate B
  (greedy_parity: 100% margin-gated / 95.8% argmax), and is published + in the
  catalog** — <https://huggingface.co/kevinqz/Qwen3-0.6B-CoreAI>. All recipes
  pass `coreai-fabric validate`, and their generated catalog entries pass the
  **live catalog's** validate + audit + io_contract + count-sync gates
  (cross-contract CI). See `docs/validation-log.md`.

  > **Namespace note.** All three publish to **your** namespace (`kevinqz`) —
  > your namespace is the source of truth. `coreai-community` is a real, separate
  > HF org (members include Hugging Face's `pcuenq`) that mirrors community CoreAI
  > conversions. You **are** a member (verify:
  > `curl -s https://huggingface.co/api/organizations/coreai-community/members | grep kevinqz`),
  > so a publish there would *succeed silently* — which is exactly why you don't
  > want it as the default: it would drop an unverified draft into a shared org
  > as the primary copy. Publish to `kevinqz` first (source of truth), then
  > mirror. fabric no longer defaults `--namespace` to a shared org.

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

Each recipe walks convert → verify → publish → register. Lead with the **int8**
(high-fidelity) tier — `qwen3-0.6b-int8` below is exactly what was published:

```bash
coreai-fabric convert  qwen3-0.6b-int8     # coreai.llm.export + int8 compression config → KV-cache asset
coreai-fabric verify   qwen3-0.6b-int8     # Gate A + Gate B (greedy_parity) BOTH pass — no escape flag
coreai-fabric publish  qwen3-0.6b-int8     # uploads to huggingface.co/kevinqz/Qwen3-0.6B-CoreAI (int8/ tier)
coreai-fabric register qwen3-0.6b-int8 --catalog-path ../coreai-catalog   # opens the catalog PR (needs gh)
```

The int8 siblings `qwen3-4b-int8` and `qwen2.5-1.5b-instruct-int8` follow the
same flow — but their parity is **not yet measured** (only Qwen3-0.6B has been
run on hardware), so `verify` them on your Mac before `publish`; don't publish an
unmeasured tier as if it were the 0.6B result. `register` needs a local clone of
`coreai-catalog` (`--catalog-path`) **and the GitHub CLI authenticated**
(`gh auth login`, Step 1); it validates the entry, replays the catalog's own CI
locally (incl. count-sync — it bumps the counts for you), and opens a PR (forking
if you lack push access). Preview without `gh` via
`coreai-fabric register <id> --dry-run`. After the PR merges:
`coreai-fabric register <id> --mark-merged`.

## Gate B is real now — and the int8 lane passes it

An earlier draft of this guide said "every production publish needs
`--allow-unverified-parity`" because Apple's `coreai.llm.eval` is a stub. That is
no longer the default reality:

- fabric ships a general **`greedy_parity`** runner (KV-cache decode, teacher-
  forced along the fp16 reference's greedy path, margin-gated, Wilson CI) that
  drives a *stateful* asset on-device. It measures **fidelity to the reference**
  — not task accuracy — and the card says exactly that.
- The **int8 lane passes it.** `qwen3-0.6b-int8` measured **100% margin-gated /
  95.8% exact-argmax / 100% top-5** on an M4 Max and is **published, live, and in
  the catalog**: <https://huggingface.co/kevinqz/Qwen3-0.6B-CoreAI>. It needed
  **no** escape flag.
- **Lead with int8 (high-fidelity), not int4.** Apple's macOS int4 preset uses
  lossy `symmetric_with_clipping` — the same Qwen3-0.6B measures only 79% argmax
  at int4. The int8 config (`quant/int8_absmax_perblock32.yaml`, absmax / no
  clipping) is ~lossless. int4 is a real *size* tier, not the default.
- The escape flags are now **exceptions**, each forcing a conscious choice:
  `--allow-unverified-parity` only for a genuine `not_run` (a metric fabric can't
  yet drive); `--publish-known-lossy-size-tier` only for a **failed** Gate B
  (e.g. int4), which the card then labels as a measured size tier. fabric never
  fakes a number and never relabels one.

So the honest default is: **convert → verify (Gate B runs for real) → publish the
tier that passes.** See `docs/parity-protocol.md` for the metric and
`docs/validation-log.md` for the measured runs.

## SotA distribution: own your namespace, mirror into `coreai-community`

`coreai-community` (huggingface.co/coreai-community) is the community org for
Apple CoreAI assets — members include Hugging Face's `pcuenq`. You **are** a
member (member #4 of 5; verify with the `curl` above), so you have write access.
Even so, the SotA move is NOT to publish straight into it — it is the pattern the
org already uses: **its repos are mirrors of individual authors' namespaces**
(their commit history literally reads "Mirror of `<author>/<model>`").

So the durable, attribution-preserving flow is:

1. **Publish to your own namespace** (`kevinqz/<model>-CoreAI`) — you own the
   canonical repo, keep attribution, and can iterate. This is Steps 1–4 above.
2. **Mirror into `coreai-community`** — since you're a member, duplicate your
   canonical repo there (HF "Duplicate this model" or the API), exactly how the
   existing entries got there.
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
- **HF Collection (in-namespace organization):** on `publish`, fabric adds the
  model to a Collection under your namespace (`publish.collection`, default
  `CoreAI · Apple on-device`) — idempotently, creating it if needed. HF
  namespaces are flat, so a Collection is the native way to **separate your
  CoreAI work from the rest of your repos** and give it one curated landing
  page. Disable per recipe by removing the field, or globally with
  `--collection ''` at scaffold time.
- **The catalog is the neutral index** that ties the three identities together:
  upstream (provenance) → your namespace (source of truth) → community mirror
  (distribution). One entry, three links, no ambiguity about who made what.

The one thing fabric does NOT do for you: the **mirror push** into
`coreai-community` — that is a one-time duplicate/API call under your account,
after you've joined the org.
