"""Hugging Face Hub API helpers.

Uses huggingface_hub when installed (pip install "coreai-fabric[hf]"),
otherwise falls back to the public REST API via urllib. All fetches are
read-only except uploads in publish.py.
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request

HF_API = "https://huggingface.co/api"
HF_RESOLVE = "https://huggingface.co"
USER_AGENT = "coreai-fabric (+https://github.com/kevinqz/coreai-fabric)"

#: Non-LFS files larger than this are not downloaded for hashing (safety valve;
#: in practice non-LFS files in model repos are small text/config files).
MAX_HASH_DOWNLOAD_BYTES = 50 * 1024 * 1024


class HFError(RuntimeError):
    pass


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise HFError(f"not found on the Hugging Face Hub: {url}") from exc
        raise HFError(f"HF API request failed ({exc.code}): {url}") from exc
    except urllib.error.URLError as exc:
        raise HFError(
            f"cannot reach the Hugging Face Hub ({exc.reason}). "
            "If offline, pass the metadata via flags instead."
        ) from exc


def model_info(repo_id: str, revision: str | None = None) -> dict:
    """Fetch model metadata: sha, license tag, pipeline_tag, size."""
    url = f"{HF_API}/models/{repo_id}"
    if revision:
        url += f"/revision/{revision}"
    info = _get_json(url)
    if not isinstance(info, dict):
        raise HFError(f"unexpected HF API response for {repo_id}")
    license_id = None
    for tag in info.get("tags", []) or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            license_id = tag.split(":", 1)[1]
            break
    return {
        "id": info.get("id"),
        "sha": info.get("sha"),
        "license": license_id,
        "pipeline_tag": info.get("pipeline_tag"),
        "size_bytes": info.get("usedStorage"),
        "gated": info.get("gated", False),
        "last_modified": info.get("lastModified"),
    }


def repo_exists(repo_id: str) -> bool:
    try:
        model_info(repo_id)
        return True
    except HFError:
        return False


def file_digests(repo_id: str, revision: str) -> list[dict]:
    """Return [{path, sha256, size_bytes}] for every file at a pinned revision.

    LFS files carry a real sha256 in the tree listing (lfs.oid). Non-LFS files
    only expose a git blob id, so they are downloaded and hashed locally —
    honest sha256 for every file, no fabricated digests.
    """
    tree = _get_json(f"{HF_API}/models/{repo_id}/tree/{revision}?recursive=true")
    if not isinstance(tree, list):
        raise HFError(f"unexpected tree response for {repo_id}@{revision}")
    digests: list[dict] = []
    for entry in tree:
        if entry.get("type") != "file":
            continue
        path = entry["path"]
        size = int(entry.get("size", 0))
        lfs = entry.get("lfs")
        if lfs and lfs.get("oid"):
            sha256 = lfs["oid"]
        else:
            if size > MAX_HASH_DOWNLOAD_BYTES:
                raise HFError(
                    f"{repo_id}@{revision}:{path} is a {size}-byte non-LFS file; "
                    "refusing to download it just to hash. Hash it manually or "
                    "store it via LFS."
                )
            sha256 = _sha256_of_url(
                f"{HF_RESOLVE}/{repo_id}/resolve/{revision}/{path}"
            )
        digests.append({"path": path, "sha256": sha256, "size_bytes": size})
    return digests


def _sha256_of_url(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    hasher = hashlib.sha256()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                hasher.update(chunk)
    except urllib.error.URLError as exc:
        raise HFError(f"failed to download {url} for hashing: {exc}") from exc
    return hasher.hexdigest()
