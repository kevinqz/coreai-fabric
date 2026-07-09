"""RFC Phase 3 (F1/F3): analyze is refusal-first — never "SOLVED", never a
coverage %, and the weight-bytes tripwire catches a hidden MoT."""
from __future__ import annotations

from coreai_fabric.analyze import (
    CANDIDATE,
    MANUAL,
    _tripwires,
    analyze,
    implied_params,
)


def test_bytes_vs_params_tripwire_fires_on_hidden_mot():
    # The SenseNova case: a textbook Qwen2 stub config against a 29.2GB checkpoint.
    info = {"size_bytes": 29_580_000_000, "gated": False}
    config = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
              "hidden_size": 3584, "num_hidden_layers": 2, "vocab_size": 152064}
    trips = _tripwires(config, info)
    assert any("weight_bytes_vs_params" in t for t in trips), trips
    result = analyze("sensnova/x", info=info, config=config)
    assert result.verdict == MANUAL
    assert any("weight_bytes_vs_params" in t for t in result.tripwires)


def test_refuses_on_auto_map():
    info = {"size_bytes": 1_000_000}
    config = {"architectures": ["XModel"], "auto_map": {"AutoModel": "modeling.XModel"}}
    result = analyze("o/x", info=info, config=config)
    assert result.verdict == MANUAL
    assert "auto_map_or_trust_remote_code" in result.tripwires


def test_refuses_on_stub_config():
    # No architectures, no model_type, AND a weight-bytes contradiction is impossible
    # to check (no dims) — the stub_config tripwire fires.
    info = {"size_bytes": 5_000_000_000}
    config = {"some_unrelated_key": True}
    result = analyze("o/x", info=info, config=config)
    assert result.verdict == MANUAL
    assert "stub_config" in result.tripwires


def test_refuses_on_gated_repo():
    info = {"size_bytes": 1_000_000, "gated": True}
    config = {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
              "hidden_size": 64, "num_hidden_layers": 2, "vocab_size": 1000}
    result = analyze("o/x", info=info, config=config)
    assert result.verdict == MANUAL
    assert "gated_repo" in result.tripwires


def test_lerobot_lane_emits_candidate_with_verbatim_shapes():
    info = {"size_bytes": 200_000_000, "gated": False}
    config = {"type": "pi0", "chunk_size": 16, "action_dim": 7, "max_state_dim": 7}
    result = analyze("lerobot/pi0_so100", info=info, config=config)
    assert result.verdict == CANDIDATE
    assert result.lane == "lerobot:pi0"
    assert result.shapes["chunk_size"] == 16  # verbatim, not invented
    assert result.shapes["action_dim"] == 7


def test_transformers_exact_match_emits_candidate():
    info = {"size_bytes": 1_000_000_000, "gated": False}
    # hidden_size large enough that the 1GB checkpoint doesn't trip the ratio.
    config = {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
              "hidden_size": 4096, "num_hidden_layers": 4, "vocab_size": 152064}
    result = analyze("o/x", info=info, config=config)
    assert result.verdict == CANDIDATE
    assert result.lane == "transformers:Qwen2ForCausalLM"


def test_unknown_architecture_still_names_a_candidate_lane():
    # Per RFC, transformers exact-match on architectures[0] names ANY architecture
    # as a candidate lane — the modeling-code verification is the gate, not a
    # fabric-side allowlist. We do not pretend to know NovelArch converts.
    info = {"size_bytes": 1_000_000}
    config = {"architectures": ["NovelArch"], "model_type": "novel",
              "hidden_size": 4096, "num_hidden_layers": 4, "vocab_size": 32000}
    result = analyze("o/x", info=info, config=config)
    assert result.verdict == CANDIDATE
    assert result.lane == "transformers:NovelArch"


def test_no_architectures_and_no_lerobot_type_is_manual():
    info = {"size_bytes": 1_000_000}
    config = {"model_type": "novel", "hidden_size": 4096, "num_hidden_layers": 4,
              "vocab_size": 32000}
    result = analyze("o/x", info=info, config=config)
    assert result.verdict == MANUAL
    assert not result.tripwires  # no tripwire fired, just no recognized lane


def test_never_emits_solved_or_coverage():
    info = {"size_bytes": 1_000_000}
    config = {"type": "act"}
    result = analyze("o/x", info=info, config=config)
    rendered = result.render()
    assert "SOLVED" not in rendered
    assert "coverage" not in rendered.lower()


def test_implied_params_returns_none_without_dims():
    assert implied_params({}) is None
    assert implied_params({"hidden_size": 64}) is None  # no num_layers
