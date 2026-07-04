---
license: {license}
base_model: {upstream_hf_repo}
{base_model_relation_line}pipeline_tag: {pipeline_tag}
library_name: coreai
{language_block}{widget_block}tags:
{tags_block}---

{mirror_line}

# {name}

**Apple Core AI chat model — runs fully on-device on Apple Silicon
(iPhone / iPad / Mac, macOS/iOS 27+).**

A quantized **stateful KV-cache chat** `.aimodel` — an Apple Core AI conversion of
[{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo}), with an embedded
tokenizer + chat template. Produced by
[coreai-fabric]({recipe_url}) and indexed by
[coreai-catalog](https://github.com/kevinqz/coreai-catalog).

## Model facts

| Field | Value |
|---|---|
{facts_block}
{variants_block}

## Use it

Install via the catalog, then run it with Apple's Foundation Models runtime:

```bash
pip install coreai-catalog && coreai-catalog install {recipe_id}
```

```swift
import CoreAILanguageModels
import FoundationModels

// modelURL = the installed macos/ bundle directory for this model
let model = try await CoreAILanguageModel(resourcesAt: modelURL)
let session = LanguageModelSession(model: model)
let reply = try await session.respond(to: "Explain on-device AI in one sentence.")
print(reply)
```

A complete, buildable example lives at
[coreai-catalog/examples/llm-chat](https://github.com/kevinqz/coreai-catalog/tree/main/examples/llm-chat).

## Requirements

- **Deployment: {min_os}, Xcode 27+.** The asset serializes with
  `minimum_os v27`, so the on-device Swift runtime requires macOS/iOS 27+.
- A Mac on **macOS 26 can convert and inspect** the asset but **cannot run** it
  on-device (the Swift runtime needs the 27 SDK).
- Apple Silicon.

## Intended use & limitations

- **Intended use:** general on-device chat / text generation. Inherits the base
  model's capabilities, languages, and biases.
- **Limitations:** {quant_caveat}. See the Evaluation section for the measured
  greedy fidelity vs the fp16 reference.

## Evaluation (parity)

- **Gate A (structure): {gate_a_status}** — the bundle's layout + metadata were
  validated on real hardware (Apple Silicon); the asset loads and generates.
{evaluation_block}
- **Runtime throughput (tok/s):** to be published once measured on the on-device
  (macOS/iOS 27) Swift runtime. Not estimated — real numbers or none.

## Provenance

| Field | Value |
|---|---|
| Base model | [{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo}) @ `{upstream_revision}` |
| Converted by | `{tool}` {tool_version} |
| Recipe | [{recipe_id}]({recipe_url}) (recipe_source: fabric) |
| Precision / quantization | {precision} / {quantization} |
| Conversion date | {date} |

Machine-readable, in this repo:
[`parity-report.json`](./parity-report.json) (gate results) ·
[`reproduce-manifest.json`](./reproduce-manifest.json) (exact tool + stack + pinned
revision to reproduce this conversion) · [`LICENSE`](./LICENSE) (upstream terms).

## License and attribution

{attribution} This artifact is a **converted + quantized derivative** of the base
model (the Apache-2.0 §4(b) change notice): weights were converted to Apple Core
AI format and quantized to {quant_label}. The conversion itself is community work.

## Links

- **Base model:** [{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo})
- **Reproduce:** [recipe `{recipe_id}`]({recipe_url}) · [runnable example](https://github.com/kevinqz/coreai-catalog/tree/main/examples/llm-chat)
- **Index:** [coreai-catalog](https://github.com/kevinqz/coreai-catalog) — the neutral registry that ties upstream ↔ this asset ↔ mirror together
{collection_link}

## The on-device Core AI ecosystem

This conversion is part of a broader open ecosystem for running models on Apple's
on-device stack — useful references if you're building here:

- [coreai-fabric](https://github.com/kevinqz/coreai-fabric) — the reproducible
  recipe → `.aimodel` pipeline that produced this asset.
- [coreai-catalog](https://github.com/kevinqz/coreai-catalog) — the index of Core
  AI models across the community, with provenance and integration snippets.
- [apple/coreai-models](https://github.com/apple/coreai-models) — Apple's official
  exporters and runtimes.
- [CoreAI Model Zoo](https://github.com/john-rocky/coreai-model-zoo) and the wider
  [coreai-community](https://huggingface.co/coreai-community) — community
  conversions across many model families.

## Not affiliated with Apple

Community conversion. Not produced, hosted, or endorsed by Apple. Apple and
Core AI are trademarks of Apple Inc., used here only to describe the target
runtime/format. This is an independent community conversion.
