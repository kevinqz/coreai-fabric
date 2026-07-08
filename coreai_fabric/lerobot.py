"""lerobot-coreai.json manifest generation (spec §14, §17.3).

When a fabric recipe has a `lerobot:` block, publish writes lerobot-coreai.json
into the HF artifact alongside parity-report.json. This file is the compatibility
manifest that lerobot-coreai reads during inspect/eval/rollout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def generate_lerobot_coreai_json(
    recipe: dict,
    parity_report: dict | None = None,
    repo_id: str | None = None,
) -> dict[str, Any]:
    """Build the lerobot-coreai.json manifest from a recipe + parity report.

    Args:
        recipe: The fabric recipe dict (must have a ``lerobot:`` block).
        parity_report: The Gate B parity report dict (from verify). May be None
            for drafts; evaluation.status will be 'not_run'.
        repo_id: The HF repo id for this artifact. Falls back to recipe id.

    Returns:
        A dict matching the lerobot-coreai.v0 schema, ready to json.dump.
    """
    lr = recipe.get("lerobot", {})
    if not lr:
        raise ValueError("recipe has no 'lerobot:' block — cannot generate lerobot-coreai.json")

    rid = recipe.get("id", "")
    artifact_repo = repo_id or rid
    upstream = recipe.get("upstream", {})
    conversion = recipe.get("conversion", {})
    action = conversion.get("action", {})
    parity = recipe.get("parity", {})

    # Evaluation block from parity report (spec §14.1).
    gate_b = (parity_report or {}).get("gate_b", {}) if parity_report else {}
    eval_status = gate_b.get("status", "not_run") if gate_b else "not_run"
    eval_metrics = gate_b.get("metrics", {}) if gate_b else {}

    evaluation: dict[str, Any] = {
        "metric": "action_parity",
        "status": eval_status,
        "n_obs": eval_metrics.get("n_obs") or parity.get("n_obs"),
        "min_chunk_cosine": eval_metrics.get("min_action_cosine") or eval_metrics.get("min_chunk_cosine"),
        "max_action_mae": eval_metrics.get("max_action_mae"),
        "max_relative_action_mae": eval_metrics.get("max_relative_action_mae"),
        "proves_numeric_fidelity": eval_status == "passed",
        "proves_task_success": False,
        "proves_robot_safety": False,
    }

    # CoreAI runtime graphs from conversion.action.graphs.
    graphs = [
        {"name": g.get("name", ""), "role": g.get("role", _infer_graph_role(g.get("name", "")))}
        for g in action.get("graphs", [])
    ]

    # Host loop from conversion.action.sampling.
    sampling = action.get("sampling", {})
    host_loop_required = sampling.get("kind") in ("flow_matching", "diffusion")
    host_loop = None
    if host_loop_required:
        host_loop = {
            "type": sampling.get("kind"),
            "solver": sampling.get("solver", "euler"),
            "num_steps": sampling.get("num_steps", 10),
        }

    # Observation/action features from conversion.action.
    obs_features = _feature_specs(action.get("observation_features") or _infer_obs_features(action))
    act_features = _feature_specs(action.get("action_features") or _infer_act_features(action))

    manifest = {
        "schema_version": "lerobot-coreai.v0",
        "runtime": "coreai",
        "framework": {
            "name": "lerobot",
            "version": lr.get("version", "0.6.0"),
            "commit": lr.get("commit"),
        },
        "policy": {
            "repo_id": artifact_repo,
            "source_repo_id": upstream.get("repo", f"lerobot/{rid}"),
            "type": lr.get("policy_type", _infer_policy_type(rid)),
            "class": lr.get("config_class"),
            "config_class": lr.get("config_class"),
        },
        "robot": {
            "type": lr.get("robot_type", _infer_robot_type(rid)),
            "action_representation": lr.get("action_representation") or action.get("action_representation"),
            "fps": action.get("fps"),
        },
        "features": {
            "observation": obs_features,
            "action": act_features,
        },
        "normalization": {
            "format": "lerobot",
            "path": "norm_stats.json",
            "sha256": None,
        },
        "coreai": {
            "artifact_format": "aimodel",
            "runner": "coreai-runner",
            "graphs": graphs,
            "host_loop_required": host_loop_required,
            **({"host_loop": host_loop} if host_loop else {}),
        },
        "evaluation": evaluation,
        "safety": {
            "default_mode": "dry_run",
            "real_actuation_requires_confirmation": True,
        },
    }

    return manifest


def write_lerobot_coreai_json(
    manifest: dict,
    output_dir: Path,
) -> Path:
    """Write the manifest as lerobot-coreai.json in output_dir.

    Returns the path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "lerobot-coreai.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return path


# --- Inference helpers ---

def _infer_policy_type(model_id: str) -> str:
    ml = model_id.lower()
    for t in ("pi0fast", "pi05", "pi0", "smolvla", "vqbet", "diffusion", "evo1", "act"):
        if t in ml:
            return t
    return "unknown"


def _infer_robot_type(model_id: str) -> str:
    ml = model_id.lower()
    for r in ("so100", "so101", "aloha", "libero"):
        if r in ml:
            return r
    return "unknown"


def _infer_graph_role(name: str) -> str:
    nl = name.lower()
    if "encode" in nl or "context" in nl:
        return "context_encoder"
    if "denoise" in nl or "action" in nl:
        return "denoise_step"
    return "unknown"


def _feature_specs(features: dict) -> dict:
    """Normalize feature specs to the manifest format."""
    result = {}
    for name, spec in features.items():
        if isinstance(spec, dict):
            result[name] = {
                "dtype": spec.get("dtype", "float32"),
                "shape": spec.get("shape"),
                "required": spec.get("required", True),
            }
        else:
            result[name] = {"dtype": "float32", "required": True}
    return result


def _infer_obs_features(action: dict) -> dict:
    """Infer observation features from action_space when not explicitly declared."""
    obs = {}
    state_dim = action.get("action_space", {}).get("max_state_dim")
    if state_dim:
        obs["observation.state"] = {"dtype": "float32", "shape": [state_dim], "required": True}
    obs["observation.images.wrist"] = {"dtype": "image", "shape": [3, 224, 224], "required": True}
    obs["task"] = {"dtype": "string", "required": False}
    return obs


def _infer_act_features(action: dict) -> dict:
    """Infer action features from action_space when not explicitly declared."""
    aSpace = action.get("action_space", {})
    dim = aSpace.get("dim") or aSpace.get("max_action_dim", 7)
    chunk = aSpace.get("chunk_size", 16)
    return {
        "action": {"dtype": "float32", "shape": [chunk, dim], "required": True}
    }
