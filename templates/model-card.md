---
license: {license}
base_model: {upstream_hf_repo}
{base_model_relation_line}pipeline_tag: {pipeline_tag}
library_name: coreai
{language_block}tags:
{tags_block}---

{mirror_line}

# {name}

A 4-bit, **stateful KV-cache chat** `.aimodel` — an Apple Core AI conversion of
[{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo}), with an embedded
tokenizer + chat template. Produced by
[coreai-fabric]({recipe_url}) and indexed by
[coreai-catalog](https://github.com/kevinqz/coreai-catalog).

## Model facts

| Field | Value |
|---|---|
{facts_block}

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
- **Limitations:** 4-bit quantized, so expect small quality deltas vs. the fp16
  base. Numeric accuracy is **not yet independently evaluated** — see Evaluation.

## Evaluation (parity)

- **Gate A (structure): {gate_a_status}** — the bundle's layout + metadata were
  validated on real hardware (Apple Silicon); the asset loads and generates.
- **Gate B (numeric accuracy): {gate_b_status}.** This is pending *upstream*, not
  skipped: the correct metric for a quantized asset is task accuracy
  (`{gate_b_metric}`), and the only conforming evaluator — Apple's
  `coreai.llm.eval` — is a stub in coreai-models 0.1.0 ("Evaluation support is
  coming soon") that cannot score a stateful KV-cache asset. It will be filled
  in when Apple ships their evaluator. fabric never fakes a parity number.

## Provenance

| Field | Value |
|---|---|
| Base model | [{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo}) @ `{upstream_revision}` |
| Converted by | `{tool}` {tool_version} |
| Recipe | [{recipe_id}]({recipe_url}) (recipe_source: fabric) |
| Precision / quantization | {precision} / {quantization} |
| Conversion date | {date} |

The machine-readable reports ship in this repo as `parity-report.json` and
`reproduce-manifest.json`.

## License and attribution

{attribution} This artifact is a **converted + quantized derivative** of the base
model (the Apache-2.0 §4(b) change notice): weights were converted to Apple Core
AI format and 4-bit quantized. The conversion itself is community work.

## Links

- [Base model](https://huggingface.co/{upstream_hf_repo}) · [Recipe]({recipe_url})
- [coreai-catalog](https://github.com/kevinqz/coreai-catalog) (neutral index)
{collection_link}

## Not affiliated with Apple

Community conversion. Not produced, hosted, or endorsed by Apple. Apple and
Core AI are trademarks of Apple Inc., used here only to describe the target
runtime/format. This is an independent community conversion.
