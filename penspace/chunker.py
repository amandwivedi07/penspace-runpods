"""Split normalized summary text into synthesizable chunks.

The model can emit at most `max_new_tokens` codec frames per call (~164s of
audio at the 2048 default), so a 10-minute summary cannot be generated in one
shot. We chunk well below that ceiling: short chunks localize QA failures and
make retries cheap, and paragraph-aware grouping lets the mastering stage place
natural pauses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

# A sentence-final period preceded by a lone capital is an initial ("J. K.
# Rowling"), not a sentence end.
_INITIAL = re.compile(r"(?:^|\s)[A-Z]\.$")
_BOUNDARY = re.compile(r"([.!?][\"')\]]*)(\s+)")


@dataclass
class Chunk:
    index: int
    text: str
    ends_paragraph: bool


def split_sentences(paragraph: str) -> List[str]:
    sentences: List[str] = []
    start = 0
    for m in _BOUNDARY.finditer(paragraph):
        candidate = paragraph[start : m.end(1)].strip()
        if not candidate or _INITIAL.search(candidate):
            continue
        sentences.append(candidate)
        start = m.end(2)
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _greedy_join(parts: List[str], sep: str, max_chars: int) -> List[str]:
    """Pack `parts` into strings of at most `max_chars`, keeping separators left."""
    keep = sep.rstrip()
    out: List[str] = []
    cur = ""
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        token = part if i == len(parts) - 1 else part + keep
        if not cur:
            cur = token
        elif len(cur) + 1 + len(token) <= max_chars:
            cur = f"{cur} {token}"
        else:
            out.append(cur)
            cur = token
    if cur:
        out.append(cur)
    return out


def _split_long_sentence(sentence: str, max_chars: int) -> List[str]:
    """Break an over-long sentence at the most natural boundary available."""
    if len(sentence) <= max_chars:
        return [sentence]
    for sep in ("; ", ", "):
        if sep in sentence:
            pieces = _greedy_join(sentence.split(sep), sep, max_chars)
            if all(len(p) <= max_chars for p in pieces):
                return pieces
    return _greedy_join(sentence.split(" "), " ", max_chars)


def chunk_text(text: str, max_chars: int = 300) -> List[Chunk]:
    """Chunk normalized text, flagging chunks that end a paragraph."""
    chunks: List[Chunk] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    for paragraph in paragraphs:
        pieces: List[str] = []
        cur = ""
        for sentence in split_sentences(paragraph):
            for piece in _split_long_sentence(sentence, max_chars):
                if not cur:
                    cur = piece
                elif len(cur) + 1 + len(piece) <= max_chars:
                    cur = f"{cur} {piece}"
                else:
                    pieces.append(cur)
                    cur = piece
        if cur:
            pieces.append(cur)

        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=piece,
                    ends_paragraph=(i == len(pieces) - 1),
                )
            )

    return chunks
