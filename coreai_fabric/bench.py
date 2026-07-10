"""`coreai-fabric bench` — the NATIVE benchmark step.

Fabric converts + verifies (fidelity), but never measured SPEED. This module
closes that loop by driving the catalog's SotA benchmark protocol/runner
against the freshly-converted bundle and distilling a DURABLE `benchmark:`
block into the recipe — so every registered model can carry REAL on-device
throughput, not a README-scraped number.

Design (mirrors the honesty discipline in verify.py):
- The measurement uses the catalog's Swift on-device runner
  (`coreai-bench-runner`, macOS 27), NOT fabric's Python Gate-B runtime
  (~0.16 tok/s — a wrong number for speed). If the runner is unavailable or
  the bundle is not an LLM/VLM, the step reports `not_run` with the exact
  reason and writes nothing. Fabric never fabricates a speed number.
- Reuse over reimplementation: the runner, protocol-config, and the
  candidate-line assembler live in coreai-catalog. This module imports the
  catalog's pure-Python helpers from a clone (the same clone `register` uses)
  and adds only the fabric-specific glue: resolving the fresh `build/<id>`
  bundle and distilling a durable, fresh-clone-survivable recipe block.
- Durability: the measured core is written back into the recipe YAML (like
  `verify._write_gate_b_measurement` writes `gate_b.measured`), because
  `build/` is gitignored — `bench-submit` must be able to re-emit the signed
  benchmarks.jsonl line from a fresh clone.

The catalog benchmark subsystem this rides on: coreai-catalog
`benchmarks/protocol-config.json` (v1.0), `bench/CoreAIBenchRunner`,
`coreai_catalog/bench.py` (assemble_benchmark_line / validate_runner_output),
`scripts/sign_benchmark.py`, and the `benchmark-validate.yml` signed lane.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

#: Default coreai-catalog clone (the one `register`/`publish --and-register`
#: auto-clone and run the catalog's own gates from). Reused so the benchmark
#: step needs no separate setup.
DEFAULT_CATALOG_CLONE = Path("~/.cache/coreai-fabric/catalog").expanduser()

#: bundle_kind values the catalog's coreai-bench-runner can measure. Protocol
#: v1.0's implemented metrics (decode_throughput, time_to_first_token) are
#: autoregressive text generation — LLM/VLM only. Everything else reports
#: not_run rather than a fabricated number (diffusion/speech/segmenter/detector
#: harnesses are declared-not-implemented upstream).
BENCHMARKABLE_KINDS = {"llm", "vlm"}

#: self_check keys carried into the durable recipe block (the subset the
#: catalog's assemble_benchmark_line consumes to grade confidence).
_SELF_CHECK_KEYS = (
    "thermal_pressure_detected",
    "all_trials_completed_requested_tokens",
    "prompt_token_count_exact",
)


class BenchError(Exception):
    """Actionable benchmark orchestration failure (message is for the user)."""


def load_catalog_bench(catalog_path: Path | str):
    """Import the catalog's pure-Python bench helpers from a clone.

    coreai-catalog is not a hard install dependency of fabric; `register`
    already works against a clone at ~/.cache/coreai-fabric/catalog. We add
    that clone to sys.path and import `coreai_catalog.bench` (the SINGLE source
    of the assembler + validators), raising an actionable error otherwise.
    """
    root = str(Path(catalog_path).expanduser())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from coreai_catalog import bench as catalog_bench  # noqa: PLC0415
    except ImportError as exc:
        raise BenchError(
            f"coreai_catalog is not importable from {root}: {exc}\n"
            "The benchmark step reuses the catalog's runner + line assembler. "
            "Point --catalog-path at a coreai-catalog clone (register auto-clones "
            "one into ~/.cache/coreai-fabric/catalog), or `pip install "
            'coreai-catalog`.'
        ) from exc
    return catalog_bench


def benchmark_precondition(recipe) -> str | None:
    """Return a not_run reason when this recipe is not benchmarkable, else None.

    The catalog's runner implements protocol v1.0's LLM metrics only. A
    non-LLM/VLM recipe is honestly skipped — never scored with a fake number.
    """
    kind = (recipe.data.get("catalog") or {}).get("bundle_kind")
    if kind in BENCHMARKABLE_KINDS:
        return None
    return (
        f"coreai-bench-runner implements the protocol v1.0 LLM metrics "
        f"(decode_throughput, time_to_first_token); bundle_kind {kind!r} is not "
        f"benchmarkable by it (llm/vlm only). Skipping — fabric fabricates no "
        f"speed number."
    )


def distill_benchmark_block(manifest: dict, observed_date: str) -> dict:
    """Run-manifest (catalog Swift runner) -> DURABLE recipe `benchmark:` block.

    Keeps only the catalog-relevant numeric core + comparability context, so
    the number survives a fresh clone (build/ is gitignored) and `bench-submit`
    can re-emit the signed benchmarks.jsonl line. Absent data stays absent —
    nothing is invented.
    """
    summaries = {m.get("metric"): m for m in manifest.get("metrics", [])}
    env = manifest.get("environment", {})
    self_check = manifest.get("self_check", {})

    block: dict = {
        "protocol_version": str(manifest.get("protocol_version", "1.0")),
        "runner_version": manifest.get("runner_version"),
        "observed_date": observed_date,
        "device_class": manifest.get("chip_family") or manifest.get("device_class"),
        "measured": {},
        "environment": {
            "os_major": str(env.get("os_major", "unknown")),
            "engine": env.get("engine_type", "unknown"),
            "compute_unit": env.get("compute_unit_inferred", "unknown"),
            "thermal_state_end": env.get("thermal_state_end", "unknown"),
            "low_power_mode": bool(env.get("low_power_mode", False)),
        },
        "self_check": {
            k: self_check[k] for k in _SELF_CHECK_KEYS if k in self_check
        },
        "warmup_runs": manifest.get("warmup_runs"),
        "measured_runs": manifest.get("measured_runs"),
    }

    for metric in ("decode_throughput", "time_to_first_token"):
        summary = summaries.get(metric)
        if summary and isinstance(summary.get("median"), (int, float)):
            entry = {"median": summary["median"]}
            if isinstance(summary.get("stddev"), (int, float)):
                entry["stddev"] = summary["stddev"]
            block["measured"][metric] = entry

    revision = manifest.get("artifact_revision")
    if revision:
        block["artifact_revision"] = revision
    return block


def manifest_from_block(
    block: dict, model_id: str, artifact_revision: str | None = None
) -> dict:
    """DURABLE recipe `benchmark:` block -> a manifest for assemble_benchmark_line.

    The inverse of :func:`distill_benchmark_block` over the fields the catalog's
    assembler consumes, so a fresh-clone `bench-submit` can rebuild the exact
    candidate line without the gitignored run-manifest. `artifact_revision`
    (the PUBLISHED HF revision, known only post-publish) can be injected here.
    """
    env = block.get("environment", {})
    measured = block.get("measured", {})
    units = {
        "decode_throughput": "tokens_per_second",
        "time_to_first_token": "milliseconds",
    }
    metrics = []
    for metric in ("decode_throughput", "time_to_first_token"):
        entry = measured.get(metric)
        if not entry:
            continue
        summary = {"metric": metric, "unit": units[metric], "median": entry.get("median")}
        if entry.get("stddev") is not None:
            summary["stddev"] = entry["stddev"]
        metrics.append(summary)

    manifest: dict = {
        "model_id": model_id,
        "protocol_version": block.get("protocol_version", "1.0"),
        "runner_version": block.get("runner_version"),
        "chip_family": block.get("device_class"),
        "warmup_runs": block.get("warmup_runs"),
        "measured_runs": block.get("measured_runs"),
        "metrics": metrics,
        "environment": {
            "os_major": env.get("os_major", "unknown"),
            "engine_type": env.get("engine", "unknown"),
            "compute_unit_inferred": env.get("compute_unit", "unknown"),
            "thermal_state_end": env.get("thermal_state_end", "unknown"),
            "low_power_mode": bool(env.get("low_power_mode", False)),
        },
        "self_check": dict(block.get("self_check", {})),
    }
    revision = artifact_revision or block.get("artifact_revision")
    if revision:
        manifest["artifact_revision"] = revision
    return manifest


def write_benchmark_measurement(recipe, block: dict) -> None:
    """Write the durable `benchmark:` block into the recipe (mirrors
    verify._write_gate_b_measurement — the durable, fresh-clone-survivable home).
    """
    recipe.data["benchmark"] = block


def format_throughput_line(block: dict | None) -> str | None:
    """Render the model-card `Runtime throughput (tok/s)` value from a durable
    benchmark block, or None when there is no measured decode number (the caller
    then keeps the honest 'to be published…' placeholder). Always tagged with
    device + compute unit + protocol version — never a bare tok/s."""
    if not block:
        return None
    decode = ((block.get("measured") or {}).get("decode_throughput") or {}).get("median")
    if not isinstance(decode, (int, float)):
        return None
    ttft = ((block.get("measured") or {}).get("time_to_first_token") or {}).get("median")
    device = block.get("device_class") or "unknown device"
    unit = (block.get("environment") or {}).get("compute_unit") or "unknown"
    protocol = block.get("protocol_version") or "?"
    runs = block.get("measured_runs")
    runner = block.get("runner_version")
    parts = [f"**{decode:.1f} tok/s** median decode"]
    if isinstance(ttft, (int, float)):
        parts.append(f"(TTFT {ttft:.0f} ms)")
    tail = f"— {device} {unit}, protocol v{protocol}"
    if runs:
        tail += f", n={runs}"
    if runner:
        tail += f", coreai-bench-runner {runner}"
    parts.append(tail)
    return (" ".join(parts)
            + ". Measured on-device (macOS/iOS 27 Swift runtime); reproduce with "
            "`coreai-fabric bench`.")


# ── Orchestration (glue; verified end-to-end, not by unit tests) ──


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_bundle_dir(bundle: Path) -> Path:
    """The dir the runner's LanguageBundle loads (descriptor with language +
    assets.main). Shared with publish via convert.resolve_bundle_dir."""
    from .convert import resolve_bundle_dir
    return resolve_bundle_dir(bundle)


def _bundle_provenance(bundle: Path, recipe, catalog_bench) -> dict:
    """Provenance for the run-context: the published HF revision (if any) + a
    digest root over the actual converted bundle bytes (catalog convention).
    Absent data stays absent — never fabricated."""
    files = [
        {"path": str(p.relative_to(bundle)), "sha256": _sha256_file(p)}
        for p in sorted(bundle.rglob("*"))
        if p.is_file()
    ]
    published = recipe.data.get("published") or {}
    return {
        "artifact_revision": published.get("revision"),
        "artifact_sha256_root": catalog_bench.compute_sha256_root(files),
        "artifact_files_total": len(files) or None,
    }


def _catalog_path(args) -> Path:
    return Path(args.catalog_path).expanduser() if getattr(args, "catalog_path", None) else DEFAULT_CATALOG_CLONE


def check_benchmark_lane_diff(changed_files: list[str], added_line_count: int) -> str | None:
    """Mirror the catalog's benchmark-validate.yml lane rule: a benchmark PR
    touches ONLY benchmarks.jsonl and adds EXACTLY one line. Returns an error
    string when violated (so bench-submit refuses before pushing), else None."""
    other = [f for f in changed_files if f and f != "benchmarks.jsonl"]
    if other:
        return ("benchmark-lane PR must touch only benchmarks.jsonl; also "
                f"changed: {', '.join(other)}")
    if added_line_count != 1:
        return f"benchmark-lane PR must add exactly 1 line; found {added_line_count}"
    return None


def _load_catalog_signer(catalog_path: Path):
    """Import `sign_entry` from the catalog's scripts/sign_benchmark.py (sigstore
    keyless). Requires the sigstore package + an OIDC identity at call time."""
    scripts = catalog_path / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    try:
        import sign_benchmark  # noqa: PLC0415
    except ImportError as exc:
        raise BenchError(
            f"cannot import sign_benchmark from {scripts}: {exc}. "
            "Install sigstore (`pip install sigstore`) and use a coreai-catalog clone."
        ) from exc
    return sign_benchmark


def _locate_runner(catalog_bench, catalog_path: Path, runner_arg: str | None):
    """Find the coreai-bench-runner binary. Prefer an explicit --runner, then
    the catalog's own discovery (COREAI_BENCH_RUNNER + classic SPM layouts),
    then the Swift-Build product layout (.build/out/Products/Release) that the
    Xcode-beta macOS-27 toolchain emits (which the catalog's locate_runner does
    not yet know). Raises the catalog's actionable BenchError if none exist."""
    import os

    if runner_arg:
        return Path(runner_arg).expanduser()
    try:
        return catalog_bench.locate_runner(catalog_path)
    except Exception:  # noqa: BLE001 — catalog BenchError; try the Swift-Build path first
        swift_build = (catalog_path / "bench" / "CoreAIBenchRunner" / ".build"
                       / "out" / "Products" / "Release" / "coreai-bench-runner")
        if swift_build.is_file() and os.access(swift_build, os.X_OK):
            return swift_build
        raise


def _model_in_catalog(catalog_path: Path, model_id: str) -> bool:
    """Whether model_id already exists in the catalog clone's catalog.yaml.
    The benchmark lane's `model_id_exists` gate requires the MODEL PR merged
    first, so bench-submit refuses until it does."""
    from .util import read_yaml

    catalog_file = catalog_path / "catalog.yaml"
    if not catalog_file.is_file():
        return False
    try:
        data = read_yaml(catalog_file)
    except Exception:  # noqa: BLE001
        return False
    return any(m.get("id") == model_id for m in (data.get("models") or []))


def cmd_bench(args) -> int:
    """Measure real on-device throughput via the catalog's Swift runner and
    write the durable `benchmark:` block into the recipe. Honest not_run
    (exit 2) when the bundle is not benchmarkable or the runner is unavailable —
    never a fabricated number."""
    from .convert import bundle_path
    from .recipes import find_recipe
    from .util import err, find_root, ok, warn, write_yaml

    root = find_root()
    recipe = find_recipe(args.id, root)

    bundle = bundle_path(root, recipe)
    if not bundle.is_dir():
        err(f"no bundle at {bundle.relative_to(root)} — run "
            f"`coreai-fabric convert {recipe.id}` first")
        return 1

    reason = benchmark_precondition(recipe)
    if reason:
        warn(f"bench not_run: {reason}")
        return 2

    catalog_path = _catalog_path(args)
    try:
        catalog_bench = load_catalog_bench(catalog_path)
    except BenchError as exc:
        warn(f"bench not_run: {exc}")
        return 2

    try:
        runner = _locate_runner(catalog_bench, catalog_path, getattr(args, "runner", None))
    except Exception as exc:  # noqa: BLE001 — catalog BenchError (macOS 27 / build guidance)
        warn(f"bench not_run: {exc}")
        return 2

    protocol_config = catalog_path / "benchmarks" / "protocol-config.json"
    if not protocol_config.is_file():
        warn(f"bench not_run: protocol config not found at {protocol_config} "
             "(is --catalog-path a coreai-catalog clone?)")
        return 2

    out_dir = Path(args.out_dir).expanduser() if getattr(args, "out_dir", None) \
        else root / "build" / recipe.id / "bench-out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The dir the runner loads is layout-dependent (fabric driver vs Apple's
    # coreai.llm.export). Resolve it by the descriptor metadata.json.
    model_dir = _resolve_bundle_dir(bundle)
    provenance = _bundle_provenance(model_dir, recipe, catalog_bench)
    nonce = catalog_bench.git_head_nonce(catalog_path)
    run_context = catalog_bench.build_run_context(recipe.id, provenance, nonce)
    run_context_path = out_dir / "run-context.json"
    run_context_path.write_text(json.dumps(run_context, indent=2) + "\n")

    # --model-path is the descriptor dir (resolved above): LanguageBundle(from:)
    # -> ModelBundle(at:) requires <path>/metadata.json with assets.main +
    # language (coreai-models ModelBundle.swift:116-119).
    print(f"  runner:   {runner}")
    print(f"  bundle:   {model_dir.relative_to(root)}")
    print(f"  protocol: {protocol_config}")
    try:
        catalog_bench.invoke_runner(
            runner=runner,
            bundle_path=str(model_dir),
            model_id=recipe.id,
            run_context_path=run_context_path,
            out_dir=out_dir,
            protocol_config_path=protocol_config,
            seed=getattr(args, "seed", 0),
        )
        manifest, trials = catalog_bench.validate_runner_output(out_dir, expected_nonce=nonce)
    except Exception as exc:  # noqa: BLE001 — catalog BenchError with runner stderr
        err(f"bench failed: {exc}")
        return 1

    block = distill_benchmark_block(manifest, observed_date=_utc_today())
    write_benchmark_measurement(recipe, block)
    write_yaml(recipe.path, recipe.data)

    decode = block["measured"].get("decode_throughput", {}).get("median")
    ttft = block["measured"].get("time_to_first_token", {}).get("median")
    ok(f"bench: {decode:.1f} tok/s decode"
       + (f" (TTFT {ttft:.0f} ms)" if ttft is not None else "")
       + f" — {block['device_class']} {block['environment']['compute_unit']}, "
       + f"protocol v{block['protocol_version']}, n={block['measured_runs']} "
       + f"({recipe.path.name})")
    print(f"  raw trials: {out_dir / 'trials.jsonl'} ({len(trials)} trials)")
    print(f"next: coreai-fabric bench-submit {recipe.id} "
          "(after the model PR is merged into the catalog)")
    return 0


def cmd_bench_submit(args) -> int:
    """Assemble the schema-valid benchmarks.jsonl line from the durable block,
    sign it (sigstore keyless), and open the signed benchmark-lane PR (one line,
    benchmarks.jsonl only). `--dry-run` stops after assemble+validate."""
    from .recipes import find_recipe
    from .util import err, find_root, ok, warn

    root = find_root()
    recipe = find_recipe(args.id, root)

    block = recipe.data.get("benchmark")
    if not block or not (block.get("measured") or {}).get("decode_throughput"):
        err(f"no durable benchmark for {recipe.id} — run "
            f"`coreai-fabric bench {recipe.id}` first")
        return 1

    catalog_path = _catalog_path(args)
    try:
        catalog_bench = load_catalog_bench(catalog_path)
    except BenchError as exc:
        err(str(exc))
        return 1

    published = recipe.data.get("published") or {}
    artifact_revision = block.get("artifact_revision") or published.get("revision")
    manifest = manifest_from_block(block, model_id=recipe.id, artifact_revision=artifact_revision)
    line = catalog_bench.assemble_benchmark_line(
        manifest, source=args.source, observed_date=block.get("observed_date"))
    errors = catalog_bench.schema_validate_line(line, catalog_path)
    if errors:
        err("assembled benchmark line failed the catalog schema:\n  " + "\n  ".join(errors))
        return 1

    ok(f"assembled benchmark line for {recipe.id}: "
       f"{line['value']} {line['unit']} on {line['device_class']} {line['compute_unit']}")
    print(json.dumps(line, ensure_ascii=False))

    # --dry-run previews the line regardless of registration (the registration
    # guard only gates a real submission).
    if getattr(args, "dry_run", False):
        warn("[dry-run] not signing or opening a PR. To submit: drop --dry-run "
             "(needs an OIDC identity — GitHub Actions ambient for auto-merge, or "
             "an interactive browser flow which lands in the catalog's curator lane).")
        return 0

    # The benchmark lane's model_id_exists gate requires the model to already be
    # in the catalog (its own PR merged) before a benchmark row can reference it.
    if not _model_in_catalog(catalog_path, recipe.id):
        err(f"model id '{recipe.id}' is not in the catalog clone yet — the "
            f"benchmark lane's model_id_exists gate requires the MODEL PR to be "
            f"merged first. Run `coreai-fabric register {recipe.id}`, let the PR "
            f"merge, `git -C {catalog_path} pull`, then bench-submit.")
        return 1

    # Sign (sigstore keyless, identity-bound). Needs an OIDC identity at call
    # time: ambient in GitHub Actions (auto-merge) or the interactive browser
    # flow (curator lane). Never a fabricated/unsigned app_benchmark_protocol row.
    signer = _load_catalog_signer(catalog_path)
    try:
        signed = signer.sign_entry(line, staging=getattr(args, "staging", False))
    except Exception as exc:  # noqa: BLE001 — sigstore raises several types
        err(f"signing failed: {exc}\n"
            "Sigstore needs an OIDC identity. Run inside GitHub Actions with "
            "`permissions: id-token: write` (ambient, auto-merge) or interactively "
            "for the browser flow. A measured app_benchmark_protocol row MUST be "
            "signed — the catalog's relay lane rejects unsigned non-upstream rows.")
        return 1
    signed_line = json.dumps(signed, sort_keys=True)
    return _apply_and_open_bench_pr(catalog_path, recipe, signed_line, args)


def _bench_pr_body(recipe, line: dict) -> str:
    return (
        f"On-device benchmark for `{recipe.id}`, measured by the catalog's "
        f"protocol v{line.get('environment', {}).get('protocol_version', '?')} runner "
        f"(`coreai-bench-runner`) via `coreai-fabric bench` and submitted by "
        f"`coreai-fabric bench-submit`.\n\n"
        f"- **{line['value']} {line['unit']}** ({line['metric']}) on "
        f"**{line['device_class']} {line['compute_unit']}**\n"
        f"- Signed (sigstore keyless, identity-bound); one line, `benchmarks.jsonl` only.\n\n"
        "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
    )


def _apply_and_open_bench_pr(catalog_path: Path, recipe, signed_line: str, args) -> int:
    """Open the signed benchmark-lane PR: touch ONLY benchmarks.jsonl, add
    EXACTLY one line (mirrors register._apply_and_open_pr + the CI lane rule)."""
    from . import CATALOG_REPO
    from .util import err, ok, warn, write_yaml

    def git(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(catalog_path), *cmd],
                              capture_output=True, text=True, check=check)

    if git("status", "--porcelain").stdout.strip():
        err(f"catalog clone at {catalog_path} has uncommitted changes — refusing "
            "to branch over them")
        return 1

    branch = f"fabric/bench-{recipe.id}"
    git("checkout", "-b", branch)

    bfile = catalog_path / "benchmarks.jsonl"
    existing = bfile.read_text()
    if not existing.endswith("\n"):
        existing += "\n"
    bfile.write_text(existing + signed_line + "\n")

    # Replay the catalog's own benchmark gates locally so the PR arrives green.
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
        tf.write(signed_line + "\n")
        line_path = tf.name
    gates = [
        (["scripts/validate_benchmark_entry.py", line_path], "validate_benchmark_entry"),
        (["scripts/physics_check.py", "--input", line_path, "--tier", "trusted"], "physics_check"),
        (["scripts/outlier_check.py", "--input", line_path, "--catalog", "benchmarks.jsonl"], "outlier_check"),
    ]
    for argv, label in gates:
        if not (catalog_path / argv[0]).exists():
            warn(f"catalog has no {argv[0]}; skipping gate '{label}'")
            continue
        proc = subprocess.run([sys.executable, *argv], cwd=catalog_path,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-6:])
            err(f"catalog benchmark gate '{label}' failed:\n{tail}")
            git("checkout", "main", check=False)
            git("branch", "-D", branch, check=False)
            return 1
        print(f"  benchmark gate {label}: ok")

    # Self-check the CI lane invariants BEFORE pushing (only benchmarks.jsonl, +1).
    changed = [f for f in git("diff", "--name-only", "HEAD").stdout.split() if f]
    added = sum(1 for ln in git("diff", "HEAD", "--", "benchmarks.jsonl").stdout.splitlines()
                if ln.startswith("+") and not ln.startswith("+++"))
    problem = check_benchmark_lane_diff(changed, added)
    if problem:
        err(f"refusing to open PR: {problem}")
        git("checkout", "main", check=False)
        git("branch", "-D", branch, check=False)
        return 1

    git("add", "benchmarks.jsonl")
    git("commit", "-m",
        f"benchmark: {recipe.id} via coreai-fabric ({line_value(signed_line)})")

    head_ref = branch
    if git("push", "-u", "origin", branch, check=False).returncode != 0:
        warn("no push access to the catalog origin — forking (third-party path)")
        subprocess.run(["gh", "repo", "fork", CATALOG_REPO, "--remote=false", "--clone=false"],
                       cwd=catalog_path, capture_output=True, text=True)
        who = subprocess.run(["gh", "api", "user", "--jq", ".login"], capture_output=True, text=True)
        fork_owner = who.stdout.strip()
        if not fork_owner:
            err("could not resolve your GitHub login for the fork")
            return 1
        git("remote", "add", "fork",
            f"https://github.com/{fork_owner}/{CATALOG_REPO.split('/')[-1]}.git", check=False)
        if git("push", "-u", "fork", branch, check=False).returncode != 0:
            err("push to fork failed")
            return 1
        head_ref = f"{fork_owner}:{branch}"

    pr = subprocess.run(
        ["gh", "pr", "create", "--repo", CATALOG_REPO, "--head", head_ref,
         "--title", f"benchmark: {recipe.id} ({args.source})",
         "--body", _bench_pr_body(recipe, json.loads(signed_line))],
        cwd=catalog_path, capture_output=True, text=True)
    if pr.returncode != 0:
        err(f"gh pr create failed:\n{pr.stderr}")
        return 1
    pr_url = pr.stdout.strip()
    print(pr_url)
    recipe.data.setdefault("benchmark", {})["pr"] = pr_url
    write_yaml(recipe.path, recipe.data)
    ok(f"benchmark-lane PR opened: {pr_url}")
    return 0


def line_value(signed_line: str) -> str:
    try:
        d = json.loads(signed_line)
        return f"{d.get('value')} {d.get('unit')}"
    except Exception:  # noqa: BLE001
        return "measured"
