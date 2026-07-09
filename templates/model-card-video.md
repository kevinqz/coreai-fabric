---
license: {license}
base_model: {upstream_hf_repo}
{base_model_relation_line}pipeline_tag: text-to-video
library_name: coreai
{gated_frontmatter}{language_block}tags:
{tags_block}---

{mirror_line}

# {name}

An Apple Core AI conversion of the **VAE decoder** from
[{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo}) — the
`AutoencoderKLWan` video autoencoder's decode path, mapping a **video latent** to
**pixel frames**. Produced by [coreai-fabric]({recipe_url}) and indexed by
[coreai-catalog](https://github.com/kevinqz/coreai-catalog).

> **VAE decoder, not the full video pipeline.** A video diffusion model is four
> separable blocks — text encoder, VAE encoder, denoising DiT, and this VAE
> decoder. This asset is ONLY the decoder (latent → pixels). The **host owns** the
> text encoder, the DiT few-step denoise loop, latent un-normalization
> (`latents_mean`/`latents_std`), frame assembly, and — for multi-chunk streaming —
> the causal `feat_cache`. It does not, by itself, generate video from a prompt.

## Model facts

| Field | Value |
|---|---|
{facts_block}

## Use it — this needs host code you supply

The bundle is a single static-size graph: `z [1,16,T,H/8,W/8]` in → `frames
[1,3,Tp,H,W]` out (spatial 8× / temporal 4× upsampling, first-chunk decode).
**You supply** the DiT denoise loop that produces the latent, the latent
un-normalization, and frame assembly in your host code (Swift or Python).

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
- This certifies the export is **numerically faithful to the source VAE decoder** — it
  does **NOT** certify end video quality. Reproduce with `coreai-fabric verify`.

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

{attribution} This artifact is a **converted derivative** of the base VAE: its
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
