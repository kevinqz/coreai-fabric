# Prefill Chunking (pf64) — Cutting iPhone Prefill from 20s to 5s

> **Source:** Zoo GLM-OCR export (2026-07-06). Pattern proven on iPhone 17 Pro.

## Problem

On iOS, models with large vision encoders (OCR, document understanding) can take **20+ seconds**
to prefill on the first inference. The bottleneck is the prefill pass over the vision encoder's
large graph — the on-device JIT compiler must specialize the full graph before any token is generated.

This makes the model effectively unusable for interactive OCR (user waits 20s before seeing any text).

## Solution: `--prefill-chunk` (pf64)

Apple's `coreai.llm.export` / `coreai.vlm.export` support a `--prefill-chunk <N>` flag that splits
the prefill into N-token chunks via a multifunction export. The zoo uses `pf64` (64-token chunks)
for GLM-OCR:

```bash
coreai.vlm.export glm-ocr --prefill-chunk 64 --output-dir build/ --output-name glm-ocr
```

### How it works

Instead of one monolithic prefill graph, the export produces:
1. A **prefill-chunk** function that processes 64 tokens at a time
2. The standard **decode** function for autoregressive generation

The host loops the prefill-chunk function `ceil(seq_len / 64)` times, feeding 64 tokens per call.
Each call is small enough to specialize quickly on-device, so the first-inference latency drops
from ~20s to ~5s (the 5s is now dominated by the decode warm-up, not prefill specialization).

### Measured impact (GLM-OCR on iPhone 17 Pro)

| Metric | Without pf64 | With pf64 |
|--------|-------------|-----------|
| First-page prefill | ~20s | ~5s |
| Throughput (pages/min) | ~3 | ~12 |
| Memory peak | Same | Same |

## When to use

- ✅ Models with large vision encoders on iOS (OCR, document understanding, VLMs with high-res images)
- ✅ Any model where first-inference latency matters more than steady-state throughput
- ❌ Small models where prefill is already fast (<2s)
- ❌ macOS (the JIT compiler is fast enough on Mac — prefill chunking adds overhead)

## Recipe integration

In a fabric recipe, add `prefill_chunk` to the conversion args:

```yaml
conversion:
  tool: coreai.vlm.export
  precision: float16
  quantization: none
  args:
    prefill-chunk: 64
```

Or for LLM OCR models:

```yaml
conversion:
  tool: coreai.llm.export
  apple_registry_name: glm-ocr
  args:
    prefill-chunk: 64
```

The `convert` command passes `--prefill-chunk 64` through verbatim.

## Related

- Zoo knowledge: `knowledge/glm-ocr-port.md`
- Apple docs: `coreai.llm.export --help` (lists `--prefill-chunk`)
- Similar pattern: chunked KV cache for MPSGraph overflow (host_cache_chunking in recipe schema)
