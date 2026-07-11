import json
from pathlib import Path

import pytest

from models.qwen3_5.spec import load_spec

FIX = Path(__file__).parent / "fixtures" / "qwythos_text_config.json"


def test_layer_types_derived():
    s = load_spec(json.loads(FIX.read_text()))
    assert s.num_layers == 32
    assert s.layer_types.count("full") == 8
    assert s.layer_types.count("linear") == 24
    assert s.layer_types[:4] == ["linear", "linear", "linear", "full"]


def test_dims_and_derived():
    s = load_spec(json.loads(FIX.read_text()))
    assert s.conv_dim == 8192            # key_dim*2 + value_dim = 2048*2 + 4096
    assert s.rotary_dim == 64            # partial_rotary_factor 0.25 * head_dim 256
    assert s.recstate_shape == (32, 128, 128)   # num_v, head_k, head_v
    assert s.image_token_id == 248056
    assert s.yarn["original_max_position_embeddings"] == 262144


def _tiny_config(layer_types=None):
    tc = {
        "model_type": "qwen3_5_text",
        "hidden_size": 8,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 16,
        "vocab_size": 100,
        "rms_norm_eps": 1e-6,
        "partial_rotary_factor": 0.5,
        "attn_output_gate": True,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 2,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_value_head_dim": 2,
        "mtp_num_hidden_layers": 1,
        "max_position_embeddings": 1024,
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": 1.0,
            "original_max_position_embeddings": 1024,
            "mrope_interleaved": True,
            "mrope_section": [1, 1, 1],
            "rope_theta": 10000,
        },
    }
    if layer_types is not None:
        tc["layer_types"] = layer_types
    return {
        "text_config": tc,
        "vision_config": {
            "model_type": "qwen3_5_vision",
            "depth": 1,
            "hidden_size": 8,
            "num_heads": 1,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 8,
            "num_position_embeddings": 4,
            "deepstack_visual_indexes": [],
        },
        "image_token_id": 1,
        "vision_start_token_id": 2,
    }


def test_derived_matches_explicit_layer_types():
    matching = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    s = load_spec(_tiny_config(layer_types=matching))
    assert s.layer_types == ["linear", "linear", "linear", "full"]

    mismatching = ["full_attention", "linear_attention", "linear_attention", "full_attention"]
    with pytest.raises(ValueError):
        load_spec(_tiny_config(layer_types=mismatching))
