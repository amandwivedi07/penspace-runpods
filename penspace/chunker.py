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
    # A paragraph short enough to be a chapter title or a "<book> by <author>"
    # line rather than prose. It is followed by a longer pause, because a
    # heading read straight into the body sounds like one run-on sentence.
    is_heading: bool = False


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


def chunk_text(
    text: str,
    max_chars: int = 300,
    heading_max_chars: int = 70,
    one_sentence_per_chunk: bool = True,
) -> List[Chunk]:
    """Chunk normalized text, flagging chunks that end a paragraph.

    The pause between chunks is the only pause a listener gets, because join()
    trims the model's own trailing silence. So packing several sentences into
    one 300 character chunk means the full stops INSIDE it are narrated with
    whatever the model happened to produce, which is often nothing: a listener
    reported a paragraph of four sentences running together, and it did,
    because those 420 characters were two chunks with one gap between them.

    one_sentence_per_chunk gives every sentence its own chunk, so every full
    stop gets sentence_gap_ms. It costs more synthesis calls for the same
    audio. Set it False to go back to packing.
    """
    chunks: List[Chunk] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    for paragraph_index, paragraph in enumerate(paragraphs):
        pieces: List[str] = []
        cur = ""
        for sentence in split_sentences(paragraph):
            for piece in _split_long_sentence(sentence, max_chars):
                if one_sentence_per_chunk:
                    # A sentence too long for one chunk still arrives here in
                    # several pieces; each becomes its own chunk, which is the
                    # old behaviour for that case and the right one — the split
                    # was made at a clause boundary.
                    pieces.append(piece)
                elif not cur:
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
                    # One short piece standing alone as its whole paragraph.
                    # A long paragraph that happens to fit in one chunk is
                    # prose, not a heading, hence the length test as well.
                    is_heading=(
                        len(pieces) == 1
                        and len(piece) <= heading_max_chars
                        and (
                            # No terminal punctuation: "Introduction".
                            not piece.rstrip().endswith((".", "!", "?"))
                            # Or it opens the text, which is where the
                            # "<book> by <author>" line lives. Later on, a
                            # short paragraph ending in a full stop is prose —
                            # "Small habits compound." is a sentence, not a
                            # heading, and pausing a second after it is worse
                            # than not pausing at all.
                            or paragraph_index < 2
                        )
                    ),
                )
            )

    return chunks
