---
license: {license}
base_model: {upstream_hf_repo}
{base_model_relation_line}pipeline_tag: robotics
library_name: coreai
{language_block}tags:
{tags_block}---

{mirror_line}

# {name}

> ⚠️ **Robot policy — needs a matching robot to actuate.** This is
> [{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo}) converted to an
> Apple Core AI `.aimodel`. Its output is a **normalized action chunk** for
> **{embodiment}**. Run it on any other robot, or with mismatched calibration /
> normalization stats, and it emits floats that *look* valid but **actuate
> garbage** — the most dangerous silent-failure mode. It is a **base checkpoint**:
> fine-tune on your robot's data before expecting useful behavior.

An Apple Core AI conversion of
[{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo}) — a **{sampling}
vision-language-action policy** (images + proprioceptive state + language → a
continuous action chunk). Produced by [coreai-fabric]({recipe_url}) and indexed by
[coreai-catalog](https://github.com/kevinqz/coreai-catalog).

## Model facts

| Field | Value |
|---|---|
{facts_block}

## Use it — this needs host code you supply

A policy is **not** a chat model: there is no stock high-level Swift runtime for
it. The bundle is the split-export shape — an `encode` graph (run once per
observation) + a `denoise_step` graph (the host drives it `num_steps` times) +
`norm_stats.json` (un-normalization). **You supply the host loop** (the N-step
sampler + un-normalization) in Swift. Recommended integration: keep LeRobot's
Python `RobotClient` for the servos/cameras/calibration, and run inference
on-device — see the `io_contract` in the catalog for the exact tensors.

```bash
pip install coreai-catalog && coreai-catalog install {recipe_id}
```

## Requirements

- **Deployment: {min_os}, Xcode 27+.** The asset serializes with `minimum_os v27`,
  so the on-device Swift runtime requires macOS/iOS 27+. A Mac on macOS 26 can
  convert and inspect it but not run it on-device.
- Apple Silicon. A matching robot to actuate (see the banner).

## Verification (action parity)

- **Gate A (structure): {gate_a_status}** — the bundle's layout + metadata were
  validated on real hardware; the graphs load.
{evaluation_block}
- This certifies the export is **numerically faithful to the source policy** — it
  does **NOT** certify real-world task success, embodiment transfer, or
  closed-loop stability. Closed-loop success-rate eval (LIBERO/ManiSkill):
  **not_run** (a separate future gate). Reproduce with `coreai-fabric verify`.

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
[`reproduce-manifest.json`](./reproduce-manifest.json) · [`LICENSE`](./LICENSE)
(upstream terms).

## License and attribution

{attribution} This artifact is a **converted derivative** of the base policy: its
weights were converted to Apple Core AI format. The conversion itself is
community work.

## Links

- **Base model:** [{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo})
- **Reproduce:** [recipe `{recipe_id}`]({recipe_url})
- **Index:** [coreai-catalog](https://github.com/kevinqz/coreai-catalog) — the
  neutral registry tying upstream ↔ this asset ↔ mirror together
{collection_link}

## The on-device Core AI ecosystem

- [coreai-fabric](https://github.com/kevinqz/coreai-fabric) — the reproducible
  recipe → `.aimodel` pipeline that produced this asset.
- [coreai-catalog](https://github.com/kevinqz/coreai-catalog) — the index of Core
  AI models across the community, with provenance and integration snippets.
- [apple/coreai-models](https://github.com/apple/coreai-models) — Apple's official
  exporters and runtimes.
- [LeRobot](https://huggingface.co/lerobot) — the upstream robotics ecosystem.

## Not affiliated with Apple

Community conversion. Not produced, hosted, or endorsed by Apple. Apple and Core
AI are trademarks of Apple Inc., used here only to describe the target
runtime/format. This is an independent community conversion.
