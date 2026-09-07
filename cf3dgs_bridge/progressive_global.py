"""progressive_global — stitched from cf3dgs_bridge/_progressive_global_parts/part*.txt (MCP size limits)."""
from __future__ import annotations
from pathlib import Path
_PART_DIR = Path(__file__).resolve().parent / "_progressive_global_parts"
_parts = sorted(_PART_DIR.glob("part*.txt"))
if not _parts:
    raise RuntimeError(f"missing source parts under {_PART_DIR}")
_CODE = "".join(p.read_text(encoding="utf-8") for p in _parts)
exec(compile(_CODE, str(Path(__file__).resolve()), "exec"), globals())
