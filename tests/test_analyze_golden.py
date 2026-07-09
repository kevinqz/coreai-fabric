"""RFC Phase 3 (F1/F3): golden retrodiction — analyze must, for each known
upstream class, EITHER refuse OR match correctly. A wrong CONFIDENT match fails.

These are OFFLINE fixtures (injected info + config, no network) standing in for
the real upstreams. The point is matcher precision, not live data — a network
marker covers the live version.
"""
from __future__ import annotations

import pytest

from coreai_fabric.analyze import CANDIDATE, MANUAL, analyze

# Each fixture: (repo, info, config, expected_verdict, expected_lane_substring or None).
# A wrong confident match (CANDIDATE when we expect MANUAL, or a wrong lane) fails.
GOLDEN = [
    # LeRobot action lanes -> candidate lane, verbatim shapes.
    ("lerobot/act_aloha_sim_transfer_cube_human",
     {"size_bytes": 210_000_000}, {"type": "act", "chunk_size": 100, "action_dim": 14},
     CANDIDATE, "lerobot:act"),
    ("lerobot/pi0_so100",
     {"size_bytes": 200_000_000}, {"type": "pi0", "chunk_size": 16, "action_dim": 7},
     CANDIDATE, "lerobot:pi0"),
    ("lerobot/diffusion_pusht",
     {"size_bytes": 1_050_864_048}, {"type": "diffusion"},
     CANDIDATE, "lerobot:diffusion"),
    # Transformers LLMs whose checkpoint size is consistent with their config -> candidate.
    # Realistic dims (intermediate_size, GQA heads) so implied_params is accurate
    # and the bytes ratio stays ~1.0 (no false tripwire on a normal model).
    ("Qwen/Qwen3-0.6B",
     {"size_bytes": 1_200_000_000},
     {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
      "hidden_size": 1024, "num_hidden_layers": 28, "intermediate_size": 3072,
      "num_attention_heads": 16, "num_key_value_heads": 8, "vocab_size": 151936,
      "tie_word_embeddings": True, "torch_dtype": "bfloat16"},
     CANDIDATE, "transformers:Qwen3ForCausalLM"),
    # THE flagship false-SOLVED case, faithful to what SenseNova ACTUALLY ships:
    # a metadata STUB config.json (no architectures / model_type / dims; the Qwen2
    # dims live in llm_config.json, which analyze does not read) -> stub_config -> MANUAL.
    ("sensenova/SenseNova-Vision-7B-MoT",
     {"size_bytes": 29_580_000_000},
     {"model_name": "SenseNova-Vision-7B-MoT", "base_model": "BAGEL-7B-MoT",
      "model_family": "unified_multimodal_generation", "parameter_size": "7B",
      "modality": ["text", "image"]},
     MANUAL, None),
    # ROBUSTNESS case (audit H3): a model that ships a REALISTIC single-backbone
    # Qwen2-7B config (28 layers, real intermediate_size) but whose checkpoint is
    # ~2x the params (a hidden MoT second expert) -> the weight-bytes tripwire must
    # fire (~1.9x >= 1.5x) -> MANUAL. This is the case a 3.0x bar let slip through.
    ("hidden/mot-7b",
     {"size_bytes": 29_200_000_000},
     {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
      "hidden_size": 3584, "num_hidden_layers": 28, "intermediate_size": 18944,
      "num_attention_heads": 28, "num_key_value_heads": 4, "vocab_size": 152064,
      "torch_dtype": "bfloat16"},
     MANUAL, None),
    # Weights-only repo (no config.json, e.g. model.pt only) -> MANUAL.
    ("weights/only", {"size_bytes": 500_000_000}, None, MANUAL, None),
    # Gated / auto_map / stub -> MANUAL.
    ("some/gated", {"size_bytes": 1_000_000, "gated": True},
     {"architectures": ["X"], "model_type": "x", "hidden_size": 64, "num_hidden_layers": 2,
      "vocab_size": 100}, MANUAL, None),
    ("trust/remote", {"size_bytes": 1_000_000},
     {"auto_map": {"AutoModel": "modeling.X"}}, MANUAL, None),
]


@pytest.mark.parametrize("repo,info,config,expected_verdict,expected_lane", GOLDEN,
                         ids=[g[0] for g in GOLDEN])
def test_golden_retrodiction(repo, info, config, expected_verdict, expected_lane):
    result = analyze(repo, info=info, config=config)
    assert result.verdict == expected_verdict, (
        f"{repo}: expected {expected_verdict!r}, got {result.verdict!r} "
        f"(tripwires={result.tripwires}, lane={result.lane})")
    if expected_lane is not None:
        assert expected_lane in (result.lane or ""), f"{repo}: expected lane ~{expected_lane!r}"


def test_golden_never_confident_wrong():
    # The load-bearing property: NO golden fixture gets a CANDIDATE we didn't expect,
    # and both the SenseNova stub AND the realistic hidden-MoT are NEVER candidates
    # (the false-SOLVED generators).
    for repo, info, config, expected_verdict, _ in GOLDEN:
        result = analyze(repo, info=info, config=config)
        if repo in ("sensenova/SenseNova-Vision-7B-MoT", "hidden/mot-7b"):
            assert result.verdict == MANUAL, f"{repo} MUST refuse (false-SOLVED generator)"


def test_hidden_mot_fires_weight_bytes_tripwire():
    # H3 regression guard: the realistic-config hidden-MoT must refuse specifically
    # via the weight-bytes tripwire (not some incidental reason), and the tripwire
    # must NOT fire on the matched normal LLM.
    repo, info, config, _, _ = next(g for g in GOLDEN if g[0] == "hidden/mot-7b")
    res = analyze(repo, info=info, config=config)
    assert res.verdict == MANUAL
    assert any("weight_bytes" in t for t in res.tripwires), res.tripwires
    qwen = next(g for g in GOLDEN if g[0] == "Qwen/Qwen3-0.6B")
    qres = analyze(qwen[0], info=qwen[1], config=qwen[2])
    assert not any("weight_bytes" in t for t in qres.tripwires), qres.tripwires
