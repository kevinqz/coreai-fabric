# /// script
# requires-python = ">=3.12"
# ///
"""Extract a standalone VLA-JEPA action-head safetensors checkpoint.

The LeRobot VLA-JEPA checkpoints place `model.action_model.*` at the front of
`model.safetensors` in one contiguous span. That lets us materialize a compact
subset checkpoint for the current export lane without carrying the Qwen or JEPA
weights around.

Usage:
  python models/vla_jepa/extract_action_head.py \
    --src build/_vla_jepa/VLA-JEPA-LIBERO/model.safetensors \
    --out build/_vla_jepa/VLA-JEPA-LIBERO/action_model.safetensors
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


PREFIX = "model.action_model."


def _read_header(src: Path) -> tuple[int, dict]:
    with src.open("rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
    return header_len, header


def _collect_action_entries(header: dict) -> list[tuple[str, dict]]:
    rows = [(k, v) for k, v in header.items() if k.startswith(PREFIX)]
    if not rows:
        raise SystemExit(f"no {PREFIX!r} tensors found")
    rows.sort(key=lambda kv: kv[1]["data_offsets"][0])
    return rows


def _rewrite_header(entries: list[tuple[str, dict]]) -> tuple[bytes, int]:
    cursor = 0
    subset = {}
    for key, meta in entries:
        start, end = meta["data_offsets"]
        size = int(end) - int(start)
        subset[key] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    blob = json.dumps(subset, separators=(",", ":")).encode("utf-8")
    return blob, cursor


def extract_action_head(src: Path, out: Path) -> None:
    header_len, header = _read_header(src)
    entries = _collect_action_entries(header)
    data_start = 8 + header_len
    first = int(entries[0][1]["data_offsets"][0])
    last = int(entries[-1][1]["data_offsets"][1])
    if first != 0:
        raise SystemExit(f"expected first action tensor at offset 0, got {first}")
    needed = data_start + last
    src_size = src.stat().st_size
    if src_size < needed:
        missing = needed - src_size
        raise SystemExit(
            f"{src} is incomplete for action-head extraction: need {needed} bytes, "
            f"have {src_size} (missing {missing})"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    header_blob, total_data = _rewrite_header(entries)
    with src.open("rb") as fin, out.open("wb") as fout:
        fout.write(struct.pack("<Q", len(header_blob)))
        fout.write(header_blob)
        fin.seek(data_start)
        remaining = total_data
        chunk = 1024 * 1024
        while remaining:
            part = fin.read(min(chunk, remaining))
            if not part:
                raise SystemExit("unexpected EOF while copying action-head bytes")
            fout.write(part)
            remaining -= len(part)

    print(
        f"ok: wrote {out} with {len(entries)} tensors "
        f"({total_data} data bytes, total file {out.stat().st_size} bytes)"
    )
    print(f"source requirement: first {needed} bytes of {src}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract VLA-JEPA action-head safetensors subset")
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    extract_action_head(args.src, args.out)


if __name__ == "__main__":
    main()
