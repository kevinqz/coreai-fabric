import torch

from models.qwen3_5.parity import state_cosine, token_match_rate


def test_state_cosine_identical_is_one():
    a = torch.randn(24, 32, 128, 128)
    assert state_cosine(a, a.clone()) > 0.9999


def test_state_cosine_detects_drift():
    a = torch.randn(24, 32, 128, 128)
    b = a + 0.5 * torch.randn_like(a)
    assert state_cosine(a, b) < 0.99


def test_token_match_rate():
    ref = [1, 2, 3, 4, 5]
    got = [1, 2, 9, 4, 5]
    assert token_match_rate(ref, got) == 0.8
