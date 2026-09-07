#!/usr/bin/env python3
"""NVIDIA-box benchmark + machine-readable report for hybrid seed → FastGS.

Stitched at runtime from ``scripts/_run_nvidia_report_parts/part*.txt`` so the
full UTF-8 source can be pushed via GitHub MCP size limits. Byte-identical to
the in-tree monolith when parts are concatenated in order.
"""
from __future__ import annotations

from pathlib import Path

_PART_DIR = Path(__file__).resolve().parent / "_run_nvidia_report_parts"
_parts = sorted(_PART_DIR.glob("part*.txt"))
if not _parts:
    raise RuntimeError(f"missing source parts under {_PART_DIR}")
_CODE = "".join(p.read_text(encoding="utf-8") for p in _parts)
exec(compile(_CODE, str(Path(__file__).resolve()), "exec"), globals())
