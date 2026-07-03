"""Fabric-native converter drivers.

Reality check (validated on real hardware, 2026-07-03, Apple M4 Max /
macOS 26.6): Apple's `coreai-torch` PyPI package (0.4.1) is a Python LIBRARY
(`TorchConverter`), not a CLI — there is no `coreai-torch` executable. The
CLI layer (`coreai.llm.export` etc.) lives in the apple/coreai-models repo,
which is NOT on PyPI and must be installed from a checkout.

These drivers give fabric a converter executable of its own
(`coreai-fabric-llm-export`) built directly on the coreai-torch public API,
speaking the same flag layout as Apple's `coreai.llm.export` so recipes can
switch tools without changing their `conversion.args`.
"""
