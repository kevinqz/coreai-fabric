"""Native benchmark step (Phase 1): fabric drives the catalog's SotA
protocol/runner and distills a DURABLE `benchmark:` block into the recipe.

Pure-Python pieces are unit-tested here with a run-manifest fixture (keys
match the catalog Swift runner, coreai-catalog/bench/CoreAIBenchRunner) — no
macOS 27 runtime needed. The catalog-integration test (distilled block ->
manifest -> assemble_benchmark_line -> schema-valid line) skips when no
coreai-catalog clone is available, mirroring the catalog's own
`skipUnless(swift)` pattern in tests/test_p1_bench.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from coreai_fabric.bench import (
    benchmark_precondition,
    distill_benchmark_block,
    load_catalog_bench,
    manifest_from_block,
    write_benchmark_measurement,
)
from coreai_fabric.recipes import Recipe, recipe_schema

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_CLONE = Path("~/.cache/coreai-fabric/catalog").expanduser()


# ── Fixtures (run-manifest shape mirrors the catalog Swift runner) ──


def make_manifest(**overrides) -> dict:
    manifest = {
        "runner_version": "0.1.0",
        "protocol_version": "1.0",
        "run_id": "fixture-run",
        "model_id": "qwen3-5-0-8b",
        "model_bundle_name": "qwen3_5_0_8b_decode_int8lin",
        "artifact_revision": "34ed8b08946395397c3b01d07d0a532237e71af3",
        "artifact_sha256_root": "a" * 64,
        "seed": 0,
        "sampling": "greedy(temperature=0)",
        "device_class": "mac-m4-max",
        "chip_family": "M4 Max",
        "prompt_tokens": 128,
        "generation_tokens": 256,
        "warmup_runs": 3,
        "measured_runs": 3,
        "metrics": [
            {"metric": "decode_throughput", "unit": "tokens_per_second",
             "median": 200.0, "stddev": 5.0, "p50": 200.0, "p95": 208.0,
             "higher_is_better": True},
            {"metric": "time_to_first_token", "unit": "milliseconds",
             "median": 90.0, "stddev": 2.0, "p50": 90.0, "p95": 93.0,
             "higher_is_better": False},
        ],
        "environment": {
            "os_version": "27.0.0", "os_major": "27", "low_power_mode": False,
            "thermal_state_start": "nominal", "thermal_state_end": "nominal",
            "engine_type": "CoreAIPipelinedEngine", "compute_unit_inferred": "GPU",
        },
        "self_check": {
            "prompt_token_count_exact": True, "greedy_sampling": True,
            "sampling_seed_applied": False, "thermal_pressure_detected": False,
            "all_trials_completed_requested_tokens": True,
            "device_class_coarsened": True,
        },
        "raw_trials_file": "trials.jsonl",
    }
    manifest.update(overrides)
    return manifest


def _recipe(bundle_kind: str | None = "llm", **catalog) -> Recipe:
    cat = {"bundle_kind": bundle_kind} if bundle_kind is not None else {}
    cat.update(catalog)
    return Recipe(Path("recipes/x.yaml"), {"id": "x", "catalog": cat})


# ── benchmark_precondition: honest not_run gate ──


def test_precondition_llm_is_benchmarkable():
    assert benchmark_precondition(_recipe("llm")) is None


def test_precondition_vlm_is_benchmarkable():
    assert benchmark_precondition(_recipe("vlm")) is None


def test_precondition_non_llm_reports_not_run_reason():
    reason = benchmark_precondition(_recipe("action"))
    assert reason is not None
    assert "action" in reason
    assert "llm" in reason.lower()


def test_precondition_missing_bundle_kind_is_not_benchmarkable():
    reason = benchmark_precondition(_recipe(bundle_kind=None))
    assert reason is not None


# ── distill_benchmark_block: run-manifest -> durable recipe block ──


def test_distill_pulls_medians_environment_and_selfcheck():
    block = distill_benchmark_block(make_manifest(), observed_date="2026-07-10")
    assert block["protocol_version"] == "1.0"
    assert block["runner_version"] == "0.1.0"
    assert block["observed_date"] == "2026-07-10"
    assert block["device_class"] == "M4 Max"
    assert block["measured"]["decode_throughput"]["median"] == 200.0
    assert block["measured"]["decode_throughput"]["stddev"] == 5.0
    assert block["measured"]["time_to_first_token"]["median"] == 90.0
    assert block["environment"]["compute_unit"] == "GPU"
    assert block["environment"]["os_major"] == "27"
    assert block["environment"]["engine"] == "CoreAIPipelinedEngine"
    assert block["environment"]["thermal_state_end"] == "nominal"
    assert block["environment"]["low_power_mode"] is False
    assert block["self_check"]["thermal_pressure_detected"] is False
    assert block["warmup_runs"] == 3
    assert block["measured_runs"] == 3


def test_distill_prefers_chip_family_over_raw_device_class():
    # chip_family ("M4 Max") is the human coarsened label the catalog rows use.
    block = distill_benchmark_block(make_manifest(chip_family="M5 Pro"),
                                    observed_date="2026-07-10")
    assert block["device_class"] == "M5 Pro"


def test_distill_omits_artifact_revision_when_absent():
    block = distill_benchmark_block(make_manifest(artifact_revision=None),
                                    observed_date="2026-07-10")
    assert "artifact_revision" not in block


def test_distill_keeps_artifact_revision_when_present():
    block = distill_benchmark_block(make_manifest(), observed_date="2026-07-10")
    assert block["artifact_revision"] == "34ed8b08946395397c3b01d07d0a532237e71af3"


# ── manifest_from_block: durable block -> manifest for assemble ──


def test_block_round_trips_to_manifest():
    block = distill_benchmark_block(make_manifest(), observed_date="2026-07-10")
    m = manifest_from_block(block, model_id="qwen3-0.6b-int8")
    assert m["model_id"] == "qwen3-0.6b-int8"
    metrics = {x["metric"]: x for x in m["metrics"]}
    assert metrics["decode_throughput"]["median"] == 200.0
    assert metrics["decode_throughput"]["stddev"] == 5.0
    assert metrics["time_to_first_token"]["median"] == 90.0
    assert m["chip_family"] == "M4 Max"
    assert m["environment"]["compute_unit_inferred"] == "GPU"
    assert m["environment"]["os_major"] == "27"
    assert m["environment"]["engine_type"] == "CoreAIPipelinedEngine"
    assert m["self_check"]["thermal_pressure_detected"] is False
    assert m["warmup_runs"] == 3
    assert m["measured_runs"] == 3
    assert m["protocol_version"] == "1.0"
    assert m["runner_version"] == "0.1.0"


def test_manifest_from_block_injects_artifact_revision_override():
    # At bench time (pre-publish) the artifact revision is unknown; bench-submit
    # (post-publish) injects the published HF revision.
    block = distill_benchmark_block(make_manifest(artifact_revision=None),
                                    observed_date="2026-07-10")
    m = manifest_from_block(block, model_id="x", artifact_revision="deadbeefcafe")
    assert m["artifact_revision"] == "deadbeefcafe"


# ── write_benchmark_measurement: durable home in the recipe ──


def test_write_benchmark_measurement_sets_block():
    r = _recipe("llm")
    block = distill_benchmark_block(make_manifest(), observed_date="2026-07-10")
    write_benchmark_measurement(r, block)
    assert r.data["benchmark"]["measured"]["decode_throughput"]["median"] == 200.0
    assert r.data["benchmark"]["device_class"] == "M4 Max"


# ── _resolve_bundle_dir: the dir the runner loads, across export layouts ──


def _write_descriptor(d: Path):
    import json as _json
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(_json.dumps(
        {"metadata_version": "0.2", "kind": "llm",
         "assets": {"main": "x.aimodel"}, "language": {"tokenizer": "t"}}))


def test_resolve_bundle_dir_fabric_layout(tmp_path):
    # Fabric driver: the descriptor lives INSIDE <id>.aimodel/.
    from coreai_fabric.bench import _resolve_bundle_dir
    bundle = tmp_path / "build" / "m" / "m.aimodel"
    _write_descriptor(bundle)
    assert _resolve_bundle_dir(bundle) == bundle


def test_resolve_bundle_dir_apple_layout(tmp_path):
    # Apple coreai.llm.export: descriptor is one level UP; <id>.aimodel is the asset.
    from coreai_fabric.bench import _resolve_bundle_dir
    outdir = tmp_path / "build" / "m"
    _write_descriptor(outdir)
    bundle = outdir / "m.aimodel"
    bundle.mkdir(parents=True)
    (bundle / "metadata.json").write_text('{"metadata_version":"0.2"}')  # asset-only meta
    assert _resolve_bundle_dir(bundle) == outdir


# ── check_benchmark_lane_diff: mirror the CI lane invariants ──


def test_lane_diff_ok_for_single_benchmark_line():
    from coreai_fabric.bench import check_benchmark_lane_diff

    assert check_benchmark_lane_diff(["benchmarks.jsonl"], 1) is None


def test_lane_diff_rejects_other_files():
    from coreai_fabric.bench import check_benchmark_lane_diff

    msg = check_benchmark_lane_diff(["benchmarks.jsonl", "catalog.yaml"], 1)
    assert msg is not None and "catalog.yaml" in msg


def test_lane_diff_rejects_wrong_line_count():
    from coreai_fabric.bench import check_benchmark_lane_diff

    assert check_benchmark_lane_diff(["benchmarks.jsonl"], 2) is not None
    assert check_benchmark_lane_diff(["benchmarks.jsonl"], 0) is not None


# ── format_throughput_line: the model-card tok/s line ──


def test_throughput_line_formats_measured_block():
    from coreai_fabric.bench import format_throughput_line

    block = distill_benchmark_block(make_manifest(), observed_date="2026-07-10")
    line = format_throughput_line(block)
    assert line is not None
    assert "200" in line and "tok/s" in line
    assert "M4 Max" in line and "GPU" in line
    assert "TTFT" in line and "90" in line
    assert "v1.0" in line  # protocol version, for comparability


def test_throughput_line_none_without_decode_measurement():
    from coreai_fabric.bench import format_throughput_line

    assert format_throughput_line(None) is None
    assert format_throughput_line({"measured": {}}) is None


# ── recipe schema: the durable benchmark block is accepted (and strict) ──


def _benchmark_subschema() -> dict:
    return recipe_schema(REPO_ROOT)["properties"]["benchmark"]


def test_recipe_schema_accepts_distilled_benchmark_block():
    from jsonschema import Draft202012Validator

    block = distill_benchmark_block(make_manifest(), observed_date="2026-07-10")
    errors = list(Draft202012Validator(_benchmark_subschema()).iter_errors(block))
    assert errors == [], errors[:1]


def test_recipe_schema_rejects_unknown_benchmark_field():
    from jsonschema import Draft202012Validator

    block = distill_benchmark_block(make_manifest(), observed_date="2026-07-10")
    block["throughput"] = 999  # additionalProperties:false must reject
    errors = list(Draft202012Validator(_benchmark_subschema()).iter_errors(block))
    assert errors


# ── Catalog integration: assembled line is schema-valid (needs a clone) ──


@pytest.mark.skipif(not CATALOG_CLONE.exists(),
                    reason="no coreai-catalog clone at ~/.cache/coreai-fabric/catalog")
def test_assembled_line_is_schema_valid_via_catalog():
    cbench = load_catalog_bench(CATALOG_CLONE)
    block = distill_benchmark_block(make_manifest(artifact_revision=None),
                                    observed_date="2026-07-10")
    m = manifest_from_block(block, model_id="qwen3-0.6b-int8",
                            artifact_revision="406aff00bcfb85863c4bc423cbf326e00ce425f1")
    line = cbench.assemble_benchmark_line(m, source="coreai-fabric",
                                          observed_date=block["observed_date"])
    assert cbench.schema_validate_line(line, CATALOG_CLONE) == []
    assert line["value"] == 200.0
    assert line["device_class"] == "M4 Max"
    assert line["compute_unit"] == "GPU"
    assert line["model_id"] == "qwen3-0.6b-int8"
    assert line["extraction_method"] == "app_benchmark_protocol"
    assert line["verification_tier"] == "unverified"
    assert line["artifact_revision"] == "406aff00bcfb85863c4bc423cbf326e00ce425f1"
