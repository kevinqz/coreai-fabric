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
    ("Qwen/Qwen3-0.6B",
     {"size_bytes": 1_200_000_000},
     {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
      "hidden_size": 1024, "num_hidden_layers": 28, "vocab_size": 151936},
     CANDIDATE, "transformers:Qwen3ForCausalLM"),
    # THE flagship false-SOLVED generator: SenseNova-Vision-7B-MoT. Its config is a
    # textbook Qwen2 stub against a 29.2GB checkpoint -> MANUAL (the bytes tripwire).
    ("sensnova/SenseNova-Vision-7B-MoT",
     {"size_bytes": 29_580_000_000},
     {"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
      "hidden_size": 3584, "num_hidden_layers": 2, "vocab_size": 152064},
     MANUAL, None),
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
    # and the hidden-MoT fixture is NEVER a candidate (the false-SOLVED generator).
    for repo, info, config, expected_verdict, _ in GOLDEN:
        result = analyze(repo, info=info, config=config)
        if repo == "sensnova/SenseNova-Vision-7B-MoT":
            assert result.verdict == MANUAL, "the hidden-MoT fixture MUST refuse"
