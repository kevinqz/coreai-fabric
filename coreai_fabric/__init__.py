"""coreai-fabric — the agent-first conversion pipeline for Apple Core AI.

Recipes in, provenance-verified .aimodel out, indexed by coreai-catalog.
Fabric never hosts weights: published artifacts go to each publisher's own
Hugging Face namespace.
"""
from __future__ import annotations

__version__ = "0.1.0"

FABRIC_REPO = "kevinqz/coreai-fabric"
FABRIC_REPO_URL = f"https://github.com/{FABRIC_REPO}"
CATALOG_REPO = "kevinqz/coreai-catalog"
CATALOG_REPO_URL = f"https://github.com/{CATALOG_REPO}"
