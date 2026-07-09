# CoreAI conversion playbook — reusable techniques

Distilled from converting the full LeRobot v0.6.0-era VLA fleet + community models to
Apple Core AI `.aimodel` assets on the ANE (macOS 27 / Xcode-27-beta, coreai_torch
0.4.1, coremltools 9.0, torch 2.9.0). Model-by-model specs live in
`v060-classb-build-specs.md` / `v060-conversion-findings.md`; this file is the portable
"how" — the traps and the fixes, ordered by how often they bite.

> **RFC Phase 4 (F4/F11):** this file IS the ACE playbook — there is **no
> `techniques.yaml`** (auto-apply has no execution vehicle: techniques are
> trace-time source edits in hand-vendored torch, and fabric refuses by design to
> drive those scripts). Each technique has an id (`T0`–`T7`) + a **framework scope
> + explicit preconditions**; a **deployability facet** (`loaded_on_ane`,
> distinct from cosine parity — see T3) is tracked because an AR generator that
> reloads N multi-GB programs per token can pass cosine while being undeployable.
> The `/reflect` ritual (`docs/reflect-ritual.md`) mines `attempts/*.jsonl` and
> proposes bounded diffs back into this file.

## Technique index (trigger-phrase → id)

| Trigger phrase (in an attempt/at error) | Technique |
|---|---|
| `complex128`, `view_as_complex`, `torch.polar` | **T1** — real-rewrite complex RoPE |
| `FoldMultiplyIntoSDPAScale`, SDPA segfault, mask-free attn | **T1** — eager attn / bool keep-mask |
| `0x10004`, `Program load failure`, dense-MoE > ~1.5GB | **T3** — graph-split + free-between |
| `_assert_tensor_metadata` on MoE einsum, fused 3D expert params | **T2** — dense-fusion + manual per-channel int8 |
| top-k routing, ragged gather, `group_gemm` | **T2** — dense-fusion MoE |
| multi-GB download, throttled CDN, only need the head | **T4** — targeted safetensors range-fetch |
| disk full, `coreai-cache` ballooned | **T4** — purge the ANE compile cache (not the HF token) |
| framework import storm (distributed/triton/flash-attn) | **T5** — standalone-import pattern |
| empty/unpopulated upstream license | **T6** — index-only, no weights path |
| N unmerged catalog PRs conflicting pairwise | **T7** — batch from fresh main |

## 0. The split discipline (the organizing principle)

Don't try to ship the whole multi-billion-param model as one ANE program. **The host
owns the big backbone; fabric ships the small deployable graph + a Gate-B fidelity
number.** Two shapes:

- **Class A — separable head.** The action/reward head is a real sub-module; ship it,
  host owns the VL/text backbone. (EVO1, Robometer, GR00T, MolmoAct2, FastWAM.)
- **Class B — coupled / whole-model.** The action denoise runs *inside* the VLM (shared
  attention / action tokens); the deployable graph is a whole-LM forward conditioned on
  the backbone's cached per-layer K/V, fed as graph inputs. (EO-1, lingbot-vla-v2.)

Either way the host owns: `embed_prefix` (the VL/text backbone → prefix K/V or embeds),
the sampling loop (flow-matching Euler / diffusion), and un-normalization. **Gate B is
`graph_output_cosine`** (non-autoregressive: cosine of the graph's raw output — velocity
/ logits / embeddings — vs the fp32 torch reference over N≥8 seeded inputs) or
`action_parity` (drive both sides through the identical sampling loop). It's a
*conversion-fidelity* metric, not task success.

## 1. coreai_torch op traps (fix before you export) — **T1**

**Framework scope:** all (any model whose torch reference uses these ops). **Precondition:** the rewrite is validated against the upstream (dense-MoE vs sparse, real-RoPE vs complex) BEFORE export — never ship an unvalidated rewrite (the "fake parity" trap where both sides share the bug).

- **complex-dtype RoPE → rewrite real.** coreai_torch has no complex dtype
  (`KeyError: torch.complex128`). Rewrite `torch.polar` / `view_as_complex` as real
  cos/sin: `(xe·cos − xo·sin, xe·sin + xo·cos)` (diff ~1.7e-6). **Also convert the
  `freqs` buffer itself to real** (`stack([f.real, f.imag], -1)`), or it re-crashes.
  Many Qwen-family / Llama RoPEs are already real cos/sin — check first; no rewrite
  needed there.
- **MPSGraph `FoldMultiplyIntoSDPAScale` segfault on mask-free SDPA (macOS 27).** A
  `F.scaled_dot_product_attention` with `attn_mask=None` segfaults at load. Two fixes,
  in order of preference:
  1. **Use eager attention** if the model has that path (manual
     `softmax(QKᵀ/√d + mask)·V`). No SDPA → the fold can't trigger. Cleanest.
  2. Inject a **data-dependent all-true bool keep-mask** when SDPA would be mask-free:
     `flag = (k.abs().sum() >= -1.0); attn_mask = flag.reshape(1,1).expand(Lq,Lk)`.
     Bool keep-mask → a `where`/select, not the foldable additive scale-multiply. A
     *constant* additive-zero mask does NOT work — it folds away and re-crashes.
- **int8 weight-only** (torchao 0.17 `Int8WeightOnlyConfig`, applied to the module
  BEFORE `torch.export`) — ~1 byte/param, needed to fit big assets on the ANE. **It
  cannot quantize 3D fused-expert `nn.Parameter`s** (MoE): `torch.export` throws
  `_assert_tensor_metadata` on the `AffineQuantizedTensor` einsum. For those, do
  **manual per-output-channel int8**: store `int8` + an fp16 scale buffer, dequantize
  (`q.to(scale.dtype) * scale`) right before the einsum.

## 2. Mixture-of-Experts export (the dense-fusion trick) — **T2**

**Framework scope:** MoE/MoT models only (lingbot-vla-v2, GR00T). **Precondition:** the bool-mask monkeypatch variant is a **lerobot artifact** (triplicated in exactly the three lerobot lanes, absent elsewhere) — triggers are framework-keyed, NOT block-keyed. Dense-fusion multiplies compute by E×, which is precisely what triggers T3.

Sparse top-k routing is export-hostile: the "which experts" gather is data-dependent
(ragged shapes) and the fast path is a `group_gemm` CUDA/triton kernel. **The only
export-friendly MoE form is DENSE:** run ALL E experts on every token, then combine with
a one-hot routing mask (the real routing weight for the top-k, zero for the rest):

```python
mask = F.one_hot(selected, E).float()            # [T, k, E]
weights = (mask * routing_weights[..., None]).sum(1)   # [T, E]  (zero off-top-k)
g = einsum('td,eid->tei', x, gate_proj)          # all experts via the fused 3D weights
u = einsum('td,eid->tei', x, up_proj)
eo = einsum('tei,edi->ted', silu(g)*u, down_proj)
out = einsum('ted,te->td', eo, weights) + shared_expert(x)
```

Mathematically **exact** vs sparse top-k (verified cosine 1.0). Cost: E× compute baked
into the graph — which is what triggers the ANE ceiling below.

## 3. The ANE program ceiling — `0x10004` (the big one) — **T3**

**Framework scope:** any large/dense graph (MoE dense-fusion output, deep stacks). **Precondition — BOUNDED host-loop pass count:** graph-split is valid *only* when the host drives a small, bounded number of programs per inference (4–10 denoise steps). An **autoregressive generator that reloads N multi-GB programs per token passes cosine parity while being UNDEPLOYABLE** — that is the **deployability facet** (`loaded_on_ane`, tracked in `gate_b.protocol`, DISTINCT from the cosine number). A green Gate B on an AR reloading loop is not a deployability claim; record `loaded_on_ane: false` honestly.

`Error … appleneuralengine … Program load failure (0x10004)` /
`could not load module from MPSGraphPackage` is a **per-program graph-complexity / size
ceiling (~1.5 GB for a dense-MoE program), NOT disk and NOT total byte-size.**

Evidence: a 36-layer × 32-expert dense-MoE fails at BOTH 3.58 GB (fp16) and 2.23 GB
(int8-experts), yet a non-MoE 4 GB int8 asset (FastWAM) loads fine. Purging the ANE
compile cache does nothing — it's structural. Mapped crossover for that graph: fp16
L=12/14 blocks load (~1.2–1.4 GB), fp16 L=18 = 1.79 GB fails; int8 L=18 = 1.11 GB /
L=24 = 1.48 GB load.

**Fix = graph-split.** Split the stack into N SEPARATE `.aimodel` programs, chain them on
the host, and **load each big program then FREE it (never co-resident).**

- **A multi-entrypoint single asset does NOT work** — `AIModel.load` instantiates all
  functions co-resident and re-crosses the ceiling. You need separate asset *files*
  loaded one at a time.
- Package as `<id>.aimodel/{metadata.json, manifest.json, programs/*.aimodel}`;
  `manifest.json` declares the chain (program → function → I/O + per-block prefix-K/V
  slices). fabric's `publish` `upload_folder(bundle)` ships it recursively, and verify's
  gate_a passes on the split `expected.bundle_files`. Driver pattern:
  `models/lingbotvla/export_split.py` (N-block).
- **Prefer fp16 blocks over int8** when they fit — int8 experts dropped the worst-obs
  cosine below the 0.999 gate; fp16 12-layer blocks fit *and* hit 0.99948.

## 4. Weight logistics — **T4**

**Framework scope:** any large checkpoint (>5GB). **Precondition:** targeted range-fetch is for a *separable* head (Class A); a Class B whole-LM forward needs the full stack, so estimate honestly before committing to range-fetch. Never purge the HF token cache.

- **Targeted safetensors range-fetch.** You usually need only the small deployable head
  (e.g. the 7.15 GB action expert), not the 17.75 GB VL backbone. Read the safetensors
  header (first 8 bytes = header len, then the JSON offset table), find the byte span of
  the tensors you need (often contiguous), and range-request only that span — 6 parallel
  chunked GETs beat a single stream on a throttled CDN. Reconstruct a clean
  `head.safetensors`. Skips a full multi-GB download.
- **Purge `~/Library/Caches/coreai-cache`** when disk gets tight mid-run — the ANE
  compile cache balloons (50 GB+). NEVER `rm -rf ~/.cache/huggingface/*` — that deletes
  the HF token.
- **Background downloads:** don't put `&` inside a `run_in_background` shell — the shell
  exits and orphans it. Let the harness background the whole command.

## 5. Standalone-import pattern — **T5**

**Framework scope:** LeRobot/transformers models with heavy dependency graphs. **Precondition:** vendor the norm/attention/decoder verbatim; rewrite ONLY what must change (MoE→dense, complex→real) and validate the rewrite — this is where a "fake parity" trap hides (both sides share the vendored bug).

To run a model's conversion in the coreai_torch venv without dragging its whole
framework (distributed / triton / flash-attn / the VL backbone): **vendor just the
needed classes** into a self-contained module (pure torch, no transformers/framework
imports). Copy the norm/attention/decoder verbatim for faithfulness; rewrite only what
must change (MoE → dense, complex RoPE → real). Validate the rewrite against the
upstream (e.g. dense-MoE vs the sparse reference) so you don't ship a "fake parity" trap
(both sides sharing the same bug). Stub `lerobot.utils.import_utils`
(`_diffusers_available=True`, `require_package=lambda *a,**k: None`) when a LeRobot
modeling file must import standalone.

## 6. Catalog / publishing process — **T6 / T7**

**Framework scope:** publishing + register. **Precondition (T6):** an empty/unpopulated upstream license is NOT an affirmative grant — index-only, no weights path, no bypass. **Precondition (T7):** batch from fresh `main`; N unmerged PRs conflict pairwise because `register` regenerates the derived surfaces.

- **Restricted or *unpopulated* upstream license → INDEX-ONLY.** An empty license field
  is NOT an affirmative redistribution grant. Ship the reproducible recipe + a measured
  Gate-B number; publish refuses the weights path (no bypass). GR00T (NVIDIA
  non-commercial) and MolmoAct2 (AI2 license unpopulated everywhere) are index-only;
  revisit MolmoAct2 if AI2 populates it.
- **Batch catalog PRs; cut from fresh `main`.** Every `register` regenerates the derived
  surfaces (`catalog.yaml`, `llms.txt`, `openapi.yaml`, `site/…`), so N unmerged PRs
  conflict *pairwise*. Consolidate multiple models into ONE PR off current `main` (the
  register helpers `_append_entry`/`_bump_artifact_count`/`generate.py`/gates compose
  cleanly), and always `git checkout main && git pull` before opening the next.
- **Sequence:** `validate → convert → verify (gate A + B) → publish (permissive only) →
  register (opens catalog PR) → merge → register --mark-merged`.

## 7. Scorecard (what this produced)

> This is the hand-built prototype. The GENERATED successor is
> `docs/scorecard.md` (`python scripts/generate_scorecard.py`), which ranks
> WITHIN a `(metric, granularity, input_protocol, reference_dtype, graph_boundary)`
> cell only — never cross-cell (RFC F2). The table below is the human-curated view.

Full LeRobot v0.6.0-era VLA fleet on the ANE:

| Model | Class | Deployable graph | Gate B (cosine) | State |
|---|---|---|---|---|
| EVO1-SO100 | A | flow-matching action head | 0.99999999999998 | published+registered |
| Robometer-4B | A | progress+success reward heads | 0.999999999996 | published+registered |
| LingBot-VA-Base | A | Wan dual-stream action DiT (int8) | 0.999999999999 | published+registered |
| FastWAM-LIBERO | A | MoT action-DiT, video-KV-cache inputs (int8) | 0.99999999999 | published+registered |
| EO-1-3B | B | whole Qwen2.5-VL LM forward (int8) | 0.99999999893 | published+registered |
| **lingbot-vla-v2-6b** | B | **36-layer MoE, 4-program graph-split (fp16)** | **0.99948** | **published+registered** |
| GR00T-N1.7-3B | A | flow-matching DiT | op-coverage proven | index-only (license) |
| MolmoAct2-LIBERO | A | separable action expert (fp16) | 0.99999681 | index-only (license) |

First Mixture-of-Experts VLA on the Apple Neural Engine (lingbot-vla-v2, via
graph-split). Every model that *can* be published is; the two that can't are index-only
purely on license.
