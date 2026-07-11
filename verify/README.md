# coreai-runtime-verify — Gate C (runtime loadability)

Fabric's **Gate B** proves *numerical parity* (the `.aimodel` computes the same
numbers as the source), but it runs on the **`coreai-core` wheel's Python
runtime** — it does **not** prove the artifact loads on the **device Swift Core
AI runtime** (iOS/macOS 27). The zoo's crash on Xcode 27 beta 3
(`Failed to convert to versioned IR`) is proof of that gap: parity-valid, yet
unloadable.

**Gate C closes it.** It loads *and runs* the artifact through the real Core AI
Swift runtime and emits a machine-checkable verdict stamped with the OS build —
turning a producer *claim* into a consumer *guarantee*.

## Build & run

Requires a Mac with the target toolchain (e.g. **Xcode 27**). Conversion (Gate A/B)
runs on any Mac; Gate C runs where the target SDK is installed.

```bash
export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
swift build -c release
.build/release/coreai-runtime-verify --model <bundle-dir-or-.aimodel> --kind llm  --input "Hi"
.build/release/coreai-runtime-verify --model <bundle-dir-or-.aimodel> --kind graph
```

- `--kind llm` loads via `CoreAILanguageModel(resourcesAt:)` + FoundationModels and generates.
- `--kind graph` loads via the system `CoreAI` framework (`AIModel`), runs the first
  function with zero-filled inputs, and reports the output shape — proving the compiled
  IR loads and executes a forward pass on this runtime.

Exit code `0` = loads+runs, `1` = failed. Verdict JSON on stdout:

```json
{
  "artifact": "<path>", "kind": "llm",
  "loads": true, "runs": true,
  "runtime": { "os": "Version 27.0 (Build 26A5378j)", "arch": "arm64" },
  "outputPreview": "Red", "elapsedSeconds": 5.97,
  "verifiedAt": "2026-07-10T22:26:49Z", "verifierVersion": "coreai-runtime-verify/0.1"
}
```

## Where it fits (the gate chain)

```
new → validate → convert (Gate A) → parity (Gate B, wheel runtime)
     → RUNTIME-LOAD (Gate C, device Swift runtime)  ← this tool
     → publish → catalog records the verdict (artifact × runtime → loads?)
```

The catalog ingests the verdict as a first-class **runtime-compatibility record**
(`data/runtime-verifications.jsonl`), so consumers can answer *"does this load on
my Xcode/OS?"* before downloading gigabytes.

## Automated driver (`gate-c.sh`)

`gate-c.sh` runs the verifier and emits a **catalog-ready** verdict line, so wiring
Gate C into the convert loop is one command per model:

```bash
# after `coreai-fabric convert <id>` produces build/<id>/<id>.aimodel:
verify/gate-c.sh <model-id> build/<id> graph \
  --append ../coreai-catalog/data/runtime-verifications.jsonl
# LLM bundles:
verify/gate-c.sh <model-id> build/<id> llm --input "Hi" --append <jsonl>
```

It builds the verifier on first use, prints the raw verdict to stderr, the
catalog line to stdout, appends it when `--append` is given, and exits non-zero
if the artifact fails to run — so it drops cleanly into CI on a Mac with the
target SDK. Verified verdicts to date: qwen2.5-0.5b (llm), whisper-large-v3-turbo,
lingbot-vision-vit-{small,base,large} (graph) — all load+run on macOS 27 (26A5378j).
