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
  | `qwen3-0.6b` | Qwen/Qwen3-0.6B | 8192 | `coreai-community/qwen3-0.6b-coreai` † |
  | `qwen2.5-1.5b-instruct` | Qwen/Qwen2.5-1.5B-Instruct | 32768 | `kevinqz/qwen2.5-1.5b-instruct-coreai` |
  | `qwen3-4b` | Qwen/Qwen3-4B | 40960 | `kevinqz/qwen3-4b-coreai` |

  All three pass `coreai-fabric validate`, and their generated catalog entries
  pass the **live catalog's** validate + audit + io_contract gates
  (cross-contract CI). `qwen3-0.6b` was additionally run end-to-end on real
  hardware — see `docs/validation-log.md`.

  † `qwen3-0.6b` is the repo's neutral reference seed, so it points at the
  community org. To publish it under **your** namespace instead, change one
  line in `recipes/qwen3-0.6b.yaml`:
  `publish.hf_target_namespace: coreai-community` → `kevinqz`. The other two
  already target `kevinqz`.

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

Three things need **your** credentials or your machine and cannot be done for
you: install the Apple toolchain, connect Hugging Face, and run the
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
coreai-fabric register qwen3-0.6b --catalog-path ../coreai-catalog   # opens the catalog PR
```

Repeat for `qwen2.5-1.5b-instruct` and `qwen3-4b`. `register` needs a local
clone of `coreai-catalog` (`--catalog-path`); it validates the entry, replays
the catalog's own CI locally, and opens a PR (forking if you lack push access).
After that PR merges: `coreai-fabric register <id> --mark-merged`.

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
