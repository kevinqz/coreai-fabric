---
license: {license}
base_model: {upstream_hf_repo}
pipeline_tag: {pipeline_tag}
tags:
- apple
- core-ai
- aimodel
- coreai-fabric
---

# {name}

`.aimodel` conversion of [{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo})
for Apple Core AI, produced by [coreai-fabric](https://github.com/kevinqz/coreai-fabric)
recipe [`{recipe_id}`]({recipe_url}).

## Provenance

| Field | Value |
|---|---|
| Base model | [{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo}) @ `{upstream_revision}` |
| Upstream license | {license} |
| Converted by | `{tool}` {tool_version} |
| Recipe | [{recipe_id}]({recipe_url}) (recipe_source: fabric) |
| Precision / quantization | {precision} / {quantization} |
| Conversion date | {date} |

## Verification gates

| Gate | Result |
|---|---|
| Gate A (bundle structure + metadata sanity) | {gate_a_status} |
| Gate B ({gate_b_metric}, threshold {gate_b_threshold}) | {gate_b_status}{gate_b_value} |

The full machine-readable reports ship in this repo as `parity-report.json`
and `conversion-manifest.json`.

## Usage

This is an Apple Core AI `.aimodel` bundle. Load it with Apple's Core AI
runtime on Apple Silicon (see the apple/coreai-models documentation). This
model is indexed by [coreai-catalog](https://github.com/kevinqz/coreai-catalog).

## Not affiliated with Apple

Community conversion. Not produced, hosted, or endorsed by Apple.
