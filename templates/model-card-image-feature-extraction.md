---
license: {license}
base_model: {upstream_hf_repo}
{base_model_relation_line}pipeline_tag: image-feature-extraction
library_name: coreai
{gated_frontmatter}{language_block}tags:
{tags_block}---

{mirror_line}

# {name}

An Apple Core AI conversion of
[{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo}) — a **vision
encoder** (ViT backbone) that maps an image to **normalized per-patch feature
tokens**, for dense downstream tasks (depth, segmentation, spatial perception).
Produced by [coreai-fabric]({recipe_url}) and indexed by
[coreai-catalog](https://github.com/kevinqz/coreai-catalog).

> **Feature backbone, not an end task.** This is a frozen encoder: it emits
> per-patch feature tokens, not depths / masks / labels. The host owns image
> preprocessing (resize to the static size, ImageNet mean/std) and any
> downstream head. Use the upstream repo for the preprocessing + task heads.

## Model facts

| Field | Value |
|---|---|
{facts_block}

## Use it — this needs host code you supply

The bundle is a single static-size graph: `image [1,3,S,S]` in → normalized
`patch_tokens [1, (S/16)^2, embed_dim]` out. **You supply** the image
preprocessing (resize to S, ImageNet normalize) and any downstream head in your
host code (Swift or Python).

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
- This certifies the export is **numerically faithful to the source backbone** — it
  does **NOT** certify downstream task accuracy. Reproduce with `coreai-fabric verify`.

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

{attribution} This artifact is a **converted derivative** of the base backbone: its
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
