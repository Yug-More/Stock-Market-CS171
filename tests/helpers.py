"""Small CSV writer for synthetic unit-test fixtures."""

from __future__ import annotations

from pathlib import Path


def write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path
