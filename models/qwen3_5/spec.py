"""Typed Qwen3.5 spec derived from a HF config dict. Single source of truth."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen35Spec:
    num_layers: int
    layer_types: list[str]
    hidden_size: int
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    conv_dim: int
    conv_kernel: int
    rotary_dim: int
    recstate_shape: tuple[int, int, int]
    image_token_id: int
    vision_start_token_id: int
    yarn: dict
    mrope_section: list[int]
    vision: dict


def _normalize_layer_type(v: str) -> str:
    if v in ("linear", "full"):
        return v
    if v == "linear_attention":
        return "linear"
    if v == "full_attention":
        return "full"
    raise ValueError(f"unrecognized layer_type value: {v!r}")


def load_spec(config: dict) -> Qwen35Spec:
    tc = config["text_config"]
    interval = tc["full_attention_interval"]
    layer_types = [
        "full" if (i + 1) % interval == 0 else "linear"
        for i in range(tc["num_hidden_layers"])
    ]

    explicit = tc.get("layer_types")
    if explicit is not None:
        normalized_explicit = [_normalize_layer_type(v) for v in explicit]
        if normalized_explicit != layer_types:
            raise ValueError(
                "derived layer_types from full_attention_interval "
                f"({layer_types}) do not match explicit config layer_types "
                f"({normalized_explicit})"
            )

    key_dim = tc["linear_key_head_dim"] * tc["linear_num_key_heads"]
    value_dim = tc["linear_value_head_dim"] * tc["linear_num_value_heads"]
    rp = tc["rope_parameters"]
    return Qwen35Spec(
        num_layers=tc["num_hidden_layers"],
        layer_types=layer_types,
        hidden_size=tc["hidden_size"],
        head_dim=tc["head_dim"],
        num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"],
        conv_dim=key_dim * 2 + value_dim,
        conv_kernel=tc["linear_conv_kernel_dim"],
        rotary_dim=int(tc["head_dim"] * tc["partial_rotary_factor"]),
        recstate_shape=(tc["linear_num_value_heads"], tc["linear_key_head_dim"], tc["linear_value_head_dim"]),
        image_token_id=config["image_token_id"],
        vision_start_token_id=config["vision_start_token_id"],
        yarn=rp,
        mrope_section=rp["mrope_section"],
        vision=config["vision_config"],
    )
