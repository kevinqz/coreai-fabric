---
license: {license}
base_model: {upstream_hf_repo}
{base_model_relation_line}pipeline_tag: robotics
library_name: coreai
{gated_frontmatter}{language_block}tags:
{tags_block}---

{mirror_line}

# {name}

An Apple Core AI conversion of
[{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo}) — the deployable
**reward-head core** of a robot-policy reward model. It maps per-frame
vision-language hidden states to **progress** (a distribution over discrete bins)
and **success** logits, for reward/progress estimation in robot learning.
Produced by [coreai-fabric]({recipe_url}) and indexed by
[coreai-catalog](https://github.com/kevinqz/coreai-catalog).

> **Reward heads, not the whole model — this needs the VLM backbone you supply.**
> Following the split discipline of the VLA lanes (EVO1 / VLA-JEPA / pi0), this
> asset ships ONLY the small MLP reward heads. The **host owns the Qwen3-VL
> backbone** (a standard VLM), the `<|prog_token|>` hidden-state extraction, and
> the decode (progress = softmax-weighted bin-mean clamped to `[0,1]`; success =
> sigmoid). Without the backbone + processor the graph is inert. This is a
> conversion-fidelity artifact, **not** a benchmarked reward signal.

## Model facts

| Field | Value |
|---|---|
{facts_block}

## Use it — this needs host code you supply

The bundle is a single static graph: per-frame hidden states
`frame_embeddings [1, T, hidden]` in → `progress_logits [1, T, bins]` +
`success_logits [1, T]` out. **You supply** the Qwen3-VL backbone that produces
those hidden states at the `<|prog_token|>` positions, plus the decode, in your
host code (Swift or Python). Use the upstream repo for the backbone + processor.

```bash
pip install coreai-catalog && coreai-catalog install {recipe_id}
```

## Requirements

- **Deployment: {min_os}, Xcode 27+.** The asset serializes with `minimum_os v27`,
  so the on-device Swift runtime requires macOS/iOS 27+. A Mac on macOS 26 can
  convert and inspect it but not run it on-device.
- Apple Silicon.
- The upstream Qwen3-VL backbone + Robometer processor (host-side) to produce the
  input hidden states.

## Verification (output parity)

- **Gate A (structure): {gate_a_status}** — the bundle's layout + metadata were
  validated; the graph loads.
{evaluation_block}
- This certifies the export is **numerically faithful to the source reward heads** —
  it does **NOT** certify reward quality or downstream task success. Reproduce with
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

{attribution} This artifact is a **converted derivative** of the base model's reward
heads: their weights were converted to Apple Core AI format. The conversion itself
is community work.{gemma_license_block}

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
