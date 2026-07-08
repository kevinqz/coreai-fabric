---
license: {license}
base_model: {upstream_hf_repo}
{base_model_relation_line}pipeline_tag: robotics
library_name: coreai
{gated_frontmatter}{language_block}tags:
{tags_block}- lerobot
- lerobot-coreai
---

{mirror_line}

# {name}

> ⚠️ **Robot policy — needs a matching robot to actuate.** This is
> [{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo}) converted to an
> Apple Core AI `.aimodel`. Its output is a **normalized action chunk** for
> **{embodiment}**. Run it on any other robot, or with mismatched calibration /
> normalization stats, and it emits floats that *look* valid but **actuate
> garbage** — the most dangerous silent-failure mode. It is a **base checkpoint**:
> fine-tune on your robot's data before expecting useful behavior.

## LeRobot CoreAI compatibility

- **Runtime:** CoreAI
- **Use with:** `lerobot-coreai`
- **Source policy framework:** LeRobot
- **LeRobot version:** {lerobot_version}
- **Policy type:** {policy_type}
- **Robot type:** {robot_type}
- **Default mode:** dry-run
- **Real actuation:** requires explicit confirmation

## What this artifact is

An Apple Core AI conversion of
[{upstream_hf_repo}](https://huggingface.co/{upstream_hf_repo}) — a **LeRobot robot policy**
that maps images + proprioceptive state (+ a language instruction, when the model
uses one) to a continuous action chunk ({sampling} sampler). Produced by
[coreai-fabric]({recipe_url}) and indexed by
[coreai-catalog](https://github.com/kevinqz/coreai-catalog).

## What it is not

- **Not proven for task success.** Action parity proves numeric fidelity, not that the
  robot achieves the task. Use dry-run, shadow, or simulation before real actuation.
- **Not proven for robot safety.** Parity does not prove safe operation on a physical robot.
- **Not a substitute for calibration.** Mismatched normalization stats or joint conventions
  produce silently wrong actions.

## How to inspect

```bash
pip install "lerobot-coreai[lerobot]"
lerobot-coreai inspect --policy.path {repo_id}
```

## How to dry-run

```bash
lerobot-coreai rollout \
  --policy.path {repo_id} \
  --robot.type {robot_type} \
  --mode dry_run
```

## Parity report

{parity_block}

## Safety note

This artifact passed numeric parity against the source LeRobot policy.
That does not prove task success or physical robot safety.
Use dry-run, shadow, or simulation before real actuation.

## Provenance

{provenance_block}
