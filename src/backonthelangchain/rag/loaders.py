"""Document loading helpers for simple local RAG examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RawDocument:
    """A raw document loaded from disk."""

    text: str
    source: str


def load_text_file(path: str | Path) -> RawDocument:
    """Load one UTF-8 text/markdown file."""

    file_path = Path(path)

    return RawDocument(
        text=file_path.read_text(encoding="utf-8"),
        source=str(file_path),
    )


def load_text_files(paths: list[str | Path]) -> list[RawDocument]:
    """Load multiple UTF-8 text/markdown files."""

    return [load_text_file(path) for path in paths]
