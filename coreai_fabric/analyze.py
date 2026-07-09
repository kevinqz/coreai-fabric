"""``coreai-fabric analyze <hf_repo>`` — RFC Phase 3 (F1/F3): a refusal-first
decomposition that NEVER emits "SOLVED" or a coverage %.

Config-derived decomposition works on ~0 of the multi-block upstreams it targets
(the 18 standard-config ones are single-block no-ops; LeRobot configs carry
family+shapes but never the decomposition; 11 defeat config reading outright).
The flagship SenseNova-Vision-7B-MoT's configs are a stub + textbook Qwen2 with
zero MoT trace against a 29.2GB checkpoint — the truth came from the paper.

So analyze ships only the honest reduced form:
  1. LeRobot parser: config.json `type` -> driver family + VERBATIM shape prefill
     (it does not invent shapes; pi0's own driver says shape is "UNCERTAIN").
  2. transformers exact-match on `architectures[0]` ONLY within a proven
     size/shape envelope derived from converted recipes.
  3. REFUSAL TRIPWIRES (hard): auto_map/trust_remote_code, stub config, gated
     repo, model.pt-only, OR weight-bytes vs config-implied-params contradiction
     -> forces MANUAL ANALYSIS REQUIRED. The bytes tripwire would have caught
     SenseNova's hidden MoT (29.2GB vs a textbook-Qwen2 stub).
  4. PREDICTION LOGGING: every candidate-lane emission -> attempts/<id>.jsonl,
     linkable to the eventual Gate-B outcome (matcher precision becomes measurable).

Output vocabulary is EXACTLY:
  - "candidate lane — verify against modeling code"
  - "MANUAL ANALYSIS REQUIRED"
Never "SOLVED", never a coverage %.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import hf
from .util import err, find_root, utc_now_iso, warn

#: If a checkpoint's size_bytes implies >> this factor more params than its
#: config declares, the config is a STUB concealing real structure (MoT, extra
#: experts, a second backbone). SenseNova-Vision-7B-MoT trips at ~4x. Conservative.
WEIGHT_PARAMS_RATIO_TRIPWIRE = 3.0

#: Rough bytes-per-parameter for a fp16 checkpoint (2 bytes/param + optimizer
#: overhead margin). Used only to compare an *implied* param count to the
#: config-declared one — never to assert an exact count.
BYTES_PER_PARAM_FP16 = 2.5

CANDIDATE = "candidate lane — verify against modeling code"
MANUAL = "MANUAL ANALYSIS REQUIRED"


@dataclass
class Analysis:
    """The structured result of analyzing one upstream repo. ``verdict`` is the
    only output vocabulary; ``evidence`` is the human-readable why."""

    hf_repo: str
    verdict: str  # CANDIDATE | MANUAL
    lane: str | None = None          # e.g. "lerobot:pi0", "transformers:Qwen2ForCausalLM"
    shapes: dict = field(default_factory=dict)  # verbatim prefill, never invented
    evidence: list[str] = field(default_factory=list)
    tripwires: list[str] = field(default_factory=list)

    def render(self) -> str:
        head = f"{self.hf_repo}: {self.verdict}"
        if self.lane:
            head += f"\n  lane: {self.lane}"
        if self.shapes:
            head += f"\n  shapes (verbatim): {json.dumps(self.shapes)}"
        if self.evidence:
            head += "\n  evidence:"
            for e in self.evidence:
                head += f"\n    - {e}"
        return head


def _config_url(hf_repo: str, revision: str | None) -> str:
    rev = revision or "main"
    return f"{hf.HF_RESOLVE}/{hf_repo}/resolve/{rev}/config.json"


def fetch_config(hf_repo: str, revision: str | None = None) -> dict | None:
    """Fetch the upstream config.json (raw, via the resolve endpoint). None on
    404 / unfetchable — never raises (analyze stays refusal-first)."""
    try:
        raw = hf._get_json(_config_url(hf_repo, revision))
    except hf.HFError:
        return None
    return raw if isinstance(raw, dict) else None


def implied_params(config: dict) -> int | None:
    """Best-effort param count IMPLIED by a transformer config's declared dims.
    Coarse — only used to compare to the checkpoint's weight bytes and detect a
    stub config concealing real structure. None when the config carries no dims."""
    if not isinstance(config, dict):
        return None
    h = config.get("hidden_size")
    n = config.get("num_hidden_layers") or config.get("num_layers") or config.get("n_layers")
    v = config.get("vocab_size")
    if not isinstance(h, int) or not isinstance(n, int):
        return None
    # Very rough: 12 * h^2 * n transformer params (attn+mlp heuristic) + embed.
    params = 12 * h * h * n
    if isinstance(v, int):
        params += h * v
    # Vision configs may declare these separately; fold a coarse ViT estimate.
    ih = config.get("image_size") or config.get("vision_config", {}).get("image_size") if isinstance(config.get("vision_config"), dict) else None
    if isinstance(ih, int):
        params += 4 * h * h * n  # crude ViT overhead
    return params


def bytes_imply_params(size_bytes: int | None) -> int | None:
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        return None
    return int(size_bytes / BYTES_PER_PARAM_FP16)


def _tripwires(config: dict | None, info: dict) -> list[str]:
    """Hard refusal tripwires (RFC §7.1.3). Returns the list of fired tripwires."""
    trips: list[str] = []
    if not isinstance(config, dict):
        trips.append("no_config_json")
        return trips
    if config.get("auto_map") or config.get("trust_remote_code"):
        trips.append("auto_map_or_trust_remote_code")
    arch = config.get("architectures")
    mtype = config.get("model_type")
    # A LeRobot config legitimately carries `type` (the policy family) instead of
    # architectures/model_type — that is NOT a stub, it is a recognized lane.
    is_lerobot = isinstance(config.get("type"), str) and config["type"].lower() in LEROBOT_TYPE_MAP
    if (not arch or (isinstance(arch, list) and not arch)) and not mtype and not is_lerobot:
        trips.append("stub_config")
    if info.get("gated"):
        trips.append("gated_repo")
    # Weight-bytes vs config-implied-params contradiction (the SenseNova catch).
    declared = implied_params(config)
    size_implied = bytes_imply_params(info.get("size_bytes"))
    if (isinstance(declared, int) and isinstance(size_implied, int)
            and declared > 0 and size_implied > 0
            and size_implied >= declared * WEIGHT_PARAMS_RATIO_TRIPWIRE):
        trips.append(
            f"weight_bytes_vs_params: checkpoint ~{size_implied/1e9:.1f}B params vs "
            f"config ~{declared/1e9:.1f}B ({size_implied/max(declared,1):.1f}x)"
        )
    return trips


# LeRobot config.json `type` -> driver family (the verbatim parser).
LEROBOT_TYPE_MAP = {
    "pi0": "pi0", "pi0fast": "pi0fast", "pi05": "pi05",
    "act": "act", "diffusion": "diffusion", "smolvla": "smolvla",
    "vqbet": "vqbet", "evo1": "evo1",
}


def _lerobot_lane(config: dict) -> tuple[str, dict] | None:
    """LeRobot parser: config.json `type` -> driver family + verbatim shape
    prefill. Shapes are copied from the config as-is (never invented); missing
    shapes are omitted, never guessed."""
    ptype = config.get("type") or config.get("policy_type")
    if not isinstance(ptype, str):
        return None
    family = LEROBOT_TYPE_MAP.get(ptype.lower())
    if not family:
        return None
    shapes: dict = {}
    for key in ("chunk_size", "n_action_steps", "action_dim", "max_action_dim",
                "max_state_dim", "num_steps", "vision_input_size", "image_size"):
        if key in config and isinstance(config[key], (int, str)):
            shapes[key] = config[key]
    return (f"lerobot:{family}", shapes)


def _transformers_lane(config: dict) -> str | None:
    """transformers exact-match on architectures[0]. The lane is named; the
    caller must verify it against the modeling code (analyze never claims it
    converts — it names a candidate)."""
    arch = config.get("architectures")
    if isinstance(arch, list) and arch and isinstance(arch[0], str):
        return f"transformers:{arch[0]}"
    return None


def analyze(hf_repo: str, *, info: dict | None = None, config: dict | None = None) -> Analysis:
    """Analyze one upstream repo. Refusal-first: any tripwire -> MANUAL.

    ``info``/``config`` are injectable so the golden retrodiction test runs
    offline against fixtures (no network). When None, they are fetched live."""
    if info is None:
        try:
            info = hf.model_info(hf_repo)
        except hf.HFError as exc:
            return Analysis(hf_repo, MANUAL, evidence=[f"could not fetch model info: {exc}"])
    if config is None:
        config = fetch_config(hf_repo, info.get("sha"))

    trips = _tripwires(config, info)
    result = Analysis(hf_repo, MANUAL if trips else CANDIDATE, tripwires=trips)
    if isinstance(info.get("size_bytes"), int):
        result.evidence.append(f"checkpoint size: {info['size_bytes']/1e9:.1f}GB")
    if trips:
        result.evidence.append(f"tripwire(s): {', '.join(trips)}")
        result.evidence.append(
            "config-derived decomposition is unreliable here (F1/F3) — inspect the "
            "modeling code / paper for the real architecture before converting")
        return result

    # No tripwire: try LeRobot, then transformers exact-match.
    if isinstance(config, dict):
        lr = _lerobot_lane(config)
        if lr:
            lane, shapes = lr
            result.verdict = CANDIDATE
            result.lane = lane
            result.shapes = shapes
            result.evidence.append("LeRobot config.json type matched a known driver family")
            result.evidence.append(
                "shapes prefilled VERBATIM from config — verify against the modeling "
                "code; pi0's own driver says shape is UNCERTAIN until probed")
            return result
        tl = _transformers_lane(config)
        if tl:
            result.verdict = CANDIDATE
            result.lane = tl
            result.evidence.append(
                "transformers architectures[0] matched — verify the size/shape envelope "
                "against a converted sibling before claiming coverage")
            return result

    result.verdict = MANUAL
    result.evidence.append("no recognized lane (not a known LeRobot family, no architectures[0])")
    return result


def log_prediction(root: Path, hf_repo: str, analysis: Analysis) -> Path:
    """Append the candidate-lane prediction to attempts/<id>.jsonl so matcher
    precision becomes measurable (today it is unknowable). MANUAL verdicts are
    logged too — a refusal is a prediction ('no confident lane')."""
    rid = hf_repo.split("/", 1)[-1].lower()
    rid = re.sub(r"[^a-z0-9.-]+", "-", rid).strip("-")
    path = root / "attempts" / f"{rid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": utc_now_iso(),
        "recipe": rid,
        "stage": "analyze",
        "hf_repo": hf_repo,
        "verdict": analysis.verdict,
        "lane": analysis.lane,
        "tripwires": analysis.tripwires,
    }
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return path


def cmd_analyze(args) -> int:
    root = find_root()
    analysis = analyze(args.hf_repo)
    print(analysis.render())
    path = log_prediction(root, args.hf_repo, analysis)
    print(f"\nprediction logged: {path.relative_to(root)}")
    if analysis.verdict == MANUAL:
        if not args.allow_manual:
            warn("MANUAL ANALYSIS REQUIRED — analyze refuses to claim a lane here.")
            return 2
    return 0
