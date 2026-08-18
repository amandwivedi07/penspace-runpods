"""Convert source documents into manifest-ready summary text.

Summaries arrive as .docx far more often than as markdown. Word files carry no
usable structure here -- every paragraph comes through styled "Normal" -- so
headings are recovered structurally: a short block with no terminal punctuation
is a section title.

Headings matter for narration because the chunker gives paragraph boundaries a
longer pause, so a recovered heading becomes an audible section break.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger(__name__)

# A heading is short and does not close like a sentence.
MAX_HEADING_CHARS = 80
_SENTENCE_TAIL = (".", "!", "?", ":", ";", ",", '"', "'")


def _is_heading(text: str) -> bool:
    if len(text) > MAX_HEADING_CHARS:
        return False
    if text.rstrip().endswith(_SENTENCE_TAIL):
        return False
    # A short line that is still clearly prose (several sentences' worth of
    # commas, say) is not a title.
    return len(text.split()) <= 12


def read_docx(path: Path) -> List[Tuple[bool, str]]:
    """Return (is_heading, text) blocks from a .docx file."""
    import docx

    document = docx.Document(str(path))
    blocks: List[Tuple[bool, str]] = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        # Word paragraphs frequently contain their own hard line breaks; treat
        # blank-line-separated runs as separate blocks.
        for part in re.split(r"\n\s*\n", text):
            part = part.strip()
            if part:
                blocks.append((_is_heading(part), part))

    return blocks


def blocks_to_markdown(blocks: List[Tuple[bool, str]]) -> str:
    lines = [f"## {text}" if heading else text for heading, text in blocks]
    return "\n\n".join(lines)


def read_document(path: Path) -> str:
    """Read .docx, .md or .txt into normalized-ish markdown text."""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return blocks_to_markdown(read_docx(path))
    if suffix in (".md", ".markdown", ".txt"):
        return path.read_text()
    raise ValueError(f"unsupported document type: {suffix}")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "summary"


def ingest(
    source: Path,
    out_dir: Path,
    summary_id: str | None = None,
    title: str | None = None,
    author: str | None = None,
    manifest: Path | None = None,
) -> dict:
    """Convert a document and append it to a manifest."""
    text = read_document(source)
    blocks = text.split("\n\n")
    headings = [b[3:] for b in blocks if b.startswith("## ")]

    summary_id = summary_id or slugify(source.stem)
    title = title or source.stem.lstrip("_").strip()

    out_dir.mkdir(parents=True, exist_ok=True)
    text_path = out_dir / f"{summary_id}.md"
    text_path.write_text(text)

    row = {"id": summary_id, "title": title}
    if author:
        row["author"] = author

    if manifest:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        # text_file is resolved relative to the manifest's own directory.
        row["text_file"] = str(
            text_path.resolve().relative_to(manifest.parent.resolve())
            if manifest.parent.resolve() in text_path.resolve().parents
            else text_path.resolve()
        )
        existing = []
        if manifest.exists():
            existing = [
                json.loads(line)
                for line in manifest.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]
        existing = [r for r in existing if r.get("id") != summary_id]
        existing.append(row)
        manifest.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in existing) + "\n"
        )

    return {
        "id": summary_id,
        "title": title,
        "text_path": str(text_path),
        "characters": len(text),
        "blocks": len(blocks),
        "headings": headings,
    }
