---
license: {license}
base_model: {upstream_hf_repo}
{base_model_relation_line}pipeline_tag: token-classification
library_name: coreai
{gated_frontmatter}{language_block}tags:
{tags_block}---

{mirror_line}

# {name}

An Apple Core AI conversion of
[{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo}) — a **token
classifier** (bidirectional encoder) that maps `(input_ids, attention_mask)` to
**per-token logits**. Produced by [coreai-fabric]({recipe_url}) and indexed by
[coreai-catalog](https://github.com/kevinqz/coreai-catalog).

> **Encoder, not a chat model.** This is a single-forward classifier — no text
> generation, no KV-cache. The host owns the **tokenizer** (use the upstream
> tokenizer at [{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo})),
> feeds token ids, and reads the per-token argmax.

## Model facts

| Field | Value |
|---|---|
{facts_block}

## Use it — this needs host code you supply

The bundle is a single static-sequence graph: `(input_ids, attention_mask)` in →
per-token `logits` out. **You supply** the upstream tokenizer and the argmax /
label mapping in your host code (Swift or Python). Token ids are int32 at the
graph boundary; pad or truncate to the static sequence length (see Model facts).

```bash
pip install coreai-catalog && coreai-catalog install {recipe_id}
```

## Requirements

- **Deployment: {min_os}, Xcode 27+.** The asset serializes with `minimum_os v27`,
  so the on-device Swift runtime requires macOS/iOS 27+. A Mac on macOS 26 can
  convert and inspect it but not run it on-device.
- Apple Silicon.

## Verification (output parity)

- **Gate A (structure): {gate_a_status}** — the bundle's layout + metadata were
  validated; the graph loads.
{evaluation_block}
- This certifies the export is **numerically faithful to the source encoder** — it
  does **NOT** certify downstream task accuracy on your data. Reproduce with
  `coreai-fabric verify`.

## Provenance

| Field | Value |
|---|---|
| Base model | [{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo}) @ `{upstream_revision}` |
| Converted by | `{tool}` {tool_version} |
| Recipe | [{recipe_id}]({recipe_url}) (recipe_source: fabric) |
| Precision / quantization | {precision} / {quantization} |
| Conversion date | {date} |

Machine-readable, in this repo:
[`parity-report.json`](./parity-report.json) ·
[`reproduce-manifest.json`](./reproduce-manifest.json) · [`LICENSE`](./LICENSE).

## License and attribution

{attribution} This artifact is a **converted derivative** of the base model: its
weights were converted to Apple Core AI format. The conversion itself is
community work.{gemma_license_block}

## Links

- **Base model:** [{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo})
- **Reproduce:** [recipe `{recipe_id}`]({recipe_url})
- **Index:** [coreai-catalog](https://github.com/kevinqz/coreai-catalog)
{collection_link}

## The on-device Core AI ecosystem

- [coreai-fabric](https://github.com/kevinqz/coreai-fabric) — the reproducible
  recipe → `.aimodel` pipeline that produced this asset.
- [coreai-catalog](https://github.com/kevinqz/coreai-catalog) — the index of Core
  AI models with provenance and integration snippets.
- [apple/coreai-models](https://github.com/apple/coreai-models) — Apple's official
  exporters and runtimes.

## Not affiliated with Apple

Community conversion. Not produced, hosted, or endorsed by Apple. Apple and Core
AI are trademarks of Apple Inc., used here only to describe the target
runtime/format.
