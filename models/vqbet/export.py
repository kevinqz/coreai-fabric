# /// script
# requires-python = ">=3.12"
# ///
"""VQ-BeT (lerobot) -> Apple .aimodel, single-graph (GREEDY / deterministic variant).

VQ-BeT maps a stack of `n_obs_steps` observations -> an action chunk in ONE forward
pass (`vqbet(batch, rollout=True)`) — structurally single-graph like ACT, NOT a
sampler loop. The catch: its rollout picks residual-VQ codes with `torch.multinomial`
(stochastic). We convert the **greedy** variant — argmax code selection — which is a
standard, valid deterministic inference mode: patch `torch.multinomial` -> argmax
during BOTH the export trace AND the parity reference, so the asset and the torch
reference are the identical deterministic function (clean action_parity). The card
says "greedy/deterministic code selection" — never implies the stochastic sampler.

TWO-VENV (like act/export.py): export in venv-A (.venv-lerobot) -> policy.pt2;
--lower in venv-B (fabric .venv, coreai_torch) -> the .aimodel.

Usage:
  .venv-lerobot/bin/python models/vqbet/export.py export --repo <hf_repo_or_local> --out build/vqbet-pusht
  .venv/bin/python         models/vqbet/export.py --lower --out build/vqbet-pusht
"""
from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

OBS_STATE = "observation.state"
OBS_IMAGES = "observation.images"


@contextlib.contextmanager
def greedy_codes():
    """Replace torch.multinomial with argmax so code selection is deterministic.
    Active during export tracing AND the parity reference — identical both sides."""
    import torch
    orig = torch.multinomial

    def _argmax(inp, num_samples=1, replacement=False, *, generator=None, out=None):
        # multinomial(probs[N,C], num_samples=1) -> [N,1]; argmax matches shape+dtype.
        return inp.argmax(dim=-1, keepdim=True).to(torch.int64)

    torch.multinomial = _argmax
    try:
        yield
    finally:
        torch.multinomial = orig


def _contract(repo: str):
    """Return (n_obs, n_cam, C, H, W, state_dim, action_chunk, action_dim) from the config."""
    from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
    policy = VQBeTPolicy.from_pretrained(repo).to("cpu").eval()
    cfg = policy.config
    img_key = next(k for k, f in cfg.input_features.items() if f.type == "VISUAL")
    C, H, W = tuple(cfg.input_features[img_key].shape)
    n_cam = sum(1 for f in cfg.input_features.values() if f.type == "VISUAL")
    state_dim = int(cfg.input_features[OBS_STATE].shape[0])
    n_obs = int(cfg.n_obs_steps)
    return policy, dict(n_obs=n_obs, n_cam=n_cam, C=C, H=H, W=W, state_dim=state_dim,
                        action_chunk=int(cfg.action_chunk_size),
                        action_dim=int(cfg.output_features["action"].shape[0]))


def _build(repo: str):
    import torch
    policy, c = _contract(repo)

    class _W(torch.nn.Module):
        def __init__(self, p): super().__init__(); self.p = p
        def forward(self, images, state):
            return self.p.vqbet({OBS_IMAGES: images, OBS_STATE: state}, rollout=True)

    return _W(policy).eval(), c


def cmd_export(repo: str, out: Path):
    import torch
    w, c = _build(repo)
    images = torch.zeros(1, c["n_obs"], c["n_cam"], c["C"], c["H"], c["W"])
    state = torch.zeros(1, c["n_obs"], c["state_dim"])
    with greedy_codes(), torch.no_grad():
        ep = torch.export.export(w, args=(images, state), strict=False)
    out.mkdir(parents=True, exist_ok=True)
    torch.export.save(ep, str(out / "policy.pt2"))
    print(f"ok: wrote {out}/policy.pt2  (images{list(images.shape)} state{list(state.shape)} "
          f"-> action_chunk[1,{c['action_chunk']},{c['action_dim']}], GREEDY)")
    print("next (venv-B): .venv/bin/python models/vqbet/export.py --lower --out", out)


def cmd_lower(out: Path):
    import torch
    from coreai_torch import TorchConverter, get_decomp_table
    ep = torch.export.load(str(out / "policy.pt2"))
    ep = ep.run_decompositions(get_decomp_table())
    conv = TorchConverter().add_exported_program(ep, output_names=["action_chunk"])
    prog = conv.to_coreai()
    prog.optimize()
    out.mkdir(parents=True, exist_ok=True)
    aimodel = out / f"{out.name}.aimodel"
    prog.save_asset(aimodel)
    print(f"ok: lowered VQ-BeT (greedy) -> {aimodel} (single deterministic graph)")


def main():
    ap = argparse.ArgumentParser(description="VQ-BeT greedy single-graph export")
    ap.add_argument("phase", nargs="?", default="export", choices=["export"])
    ap.add_argument("--repo", default="lerobot/vqbet_pusht")
    ap.add_argument("--lower", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cmd_lower(args.out) if args.lower else cmd_export(args.repo, args.out)


if __name__ == "__main__":
    main()
