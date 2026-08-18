"""Text normalization for TTS narration.

Book summaries carry a lot of things a TTS model reads badly: markdown syntax,
citation markers, abbreviations, years, currency and percentages. This module
rewrites them into plain spoken English *before* chunking, so the sentence
splitter also benefits (expanding "Dr." removes a false sentence boundary).

Paragraph breaks are preserved as blank lines; the chunker needs them to decide
where to place longer pauses.
"""

from __future__ import annotations

import re

from num2words import num2words

# Numbers in this range are spoken as years ("nineteen eighty-four") rather than
# cardinals ("one thousand nine hundred eighty-four"). num2words' year mode
# produces natural output across the whole range, so the occasional non-year
# (e.g. "1500 calories" -> "fifteen hundred calories") still reads acceptably.
YEAR_MIN, YEAR_MAX = 1100, 2099

# Expanded before sentence splitting so their periods stop looking like sentence
# ends. Ordered longest-first where prefixes overlap.
ABBREVIATIONS = [
    (r"\be\.g\.", "for example"),
    (r"\bi\.e\.", "that is"),
    (r"\betc\.", "et cetera"),
    (r"\bvs\.?\b", "versus"),
    (r"\bcf\.", "compare"),
    (r"\bapprox\.", "approximately"),
    (r"\best\.\s*(?=\d)", "estimated "),
    (r"\bMrs\.", "Missus"),
    (r"\bMr\.", "Mister"),
    (r"\bMs\.", "Miss"),
    (r"\bDr\.", "Doctor"),
    (r"\bProf\.", "Professor"),
    (r"\bSt\.", "Saint"),
    (r"\bCh\.\s*(?=\d)", "Chapter "),
    (r"\bpp\.\s*(?=\d)", "pages "),
    (r"\bp\.\s*(?=\d)", "page "),
    (r"\bNo\.\s*(?=\d)", "Number "),
    (r"\bFig\.\s*(?=\d)", "Figure "),
    (r"\bInc\.", "Incorporated"),
    (r"\bLtd\.", "Limited"),
    (r"\bCo\.", "Company"),
    (r"\bU\.S\.A\.", "USA"),
    (r"\bU\.S\.", "US"),
    (r"\bU\.K\.", "UK"),
]

SCALE_WORDS = r"(?:hundred|thousand|million|billion|trillion)"

# Headings and bullets are unambiguous block starts. A "12." line is not: it is
# equally likely to be prose that soft-wrapped onto a number, so it needs
# context (see unwrap_soft_breaks).
# Indentation classes below are [ \t], never \s. Under re.MULTILINE, `^` matches
# right after a newline, so a leading `\s{0,3}` can swallow the *next* newline --
# which silently deletes the blank line separating two paragraphs and merges
# them. Keep newlines out of every line-anchored pattern.
_HEADING_OR_BULLET = re.compile(r"^[ \t]{0,3}(?:#{1,6}[ \t]|[-*+][ \t])")
_ORDERED_ITEM = re.compile(r"^[ \t]{0,3}\d+[.)][ \t]")
_SENTENCE_END = re.compile(r"[.!?:][\"')\]]*$")

_MD_CODE_FENCE = re.compile(r"```.*?```", re.S)
_MD_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.M)
_MD_BLOCKQUOTE = re.compile(r"^[ \t]{0,3}>[ \t]?", re.M)
_MD_HRULE = re.compile(r"^[ \t]{0,3}(?:[-*_][ \t]*){3,}$", re.M)
_MD_LIST = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+", re.M)
_MD_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3})(\S.*?\S|\S)\1", re.S)
_CITATION = re.compile(r"\[\s*\d+(?:\s*[,-]\s*\d+)*\s*\]")
_FOOTNOTE = re.compile(r"\[\^[^\]]+\]")


def unwrap_soft_breaks(text: str) -> str:
    """Join soft-wrapped lines, promoting headings and list items to blocks.

    This must run before list markers are stripped. Otherwise a sentence that
    happens to wrap onto a line beginning with a number and a period -- "...when
    the anchor is\\n35. Availability:" -- looks exactly like an ordered list
    item, and the "35." is silently deleted.
    """
    blocks = re.split(r"\n\s*\n", text)
    items: list[str] = []

    for block in blocks:
        current: list[str] = []
        in_ordered_list = False

        for raw_line in block.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            if not current:
                starts_block = True
                in_ordered_list = bool(_ORDERED_ITEM.match(line))
            elif _HEADING_OR_BULLET.match(line):
                starts_block = True
                in_ordered_list = False
            elif _ORDERED_ITEM.match(line):
                # A real list item only if we are already in a numbered list, or
                # the previous line actually closed a sentence. Otherwise this
                # is prose that wrapped onto a number.
                starts_block = in_ordered_list or bool(
                    _SENTENCE_END.search(current[-1])
                )
                in_ordered_list = starts_block
            else:
                starts_block = False

            if starts_block:
                current.append(line)
            else:
                current[-1] = f"{current[-1]} {line}"

        items.extend(current)

    return "\n\n".join(items)


def strip_markdown(text: str) -> str:
    text = _MD_IMAGE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_INLINE_CODE.sub(r"\1", text)
    text = _MD_HRULE.sub("", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_LIST.sub("", text)
    text = _MD_EMPHASIS.sub(r"\2", text)
    return text


def expand_abbreviations(text: str) -> str:
    for pattern, replacement in ABBREVIATIONS:
        text = re.sub(pattern, replacement, text)
    return text


def _cardinal(n: str) -> str:
    """Spell a numeric string, handling decimals and thousands separators."""
    n = n.replace(",", "")
    if "." in n:
        return num2words(float(n))
    return num2words(int(n))


def _year(n: int) -> str:
    return num2words(n, to="year")


def _sub_currency(text: str) -> str:
    """"$5 million" -> "five million dollars"; "$1" -> "one dollar"."""

    def repl(m):
        amount, scale = m.group(1), m.group(2)
        spoken = _cardinal(amount)
        if scale:
            return f"{spoken} {scale.lower()} dollars"
        value = float(amount.replace(",", ""))
        return f"{spoken} {'dollar' if value == 1 else 'dollars'}"

    # The scale word's leading whitespace lives inside the optional group, so a
    # non-match leaves the following space intact ("$150 or", not "dollarsor").
    return re.sub(
        rf"\$\s*(\d[\d,]*(?:\.\d+)?)(?:\s+({SCALE_WORDS}))?",
        repl,
        text,
        flags=re.I,
    )


def _sub_percent(text: str) -> str:
    return re.sub(
        r"(\d[\d,]*(?:\.\d+)?)\s*%",
        lambda m: f"{_cardinal(m.group(1))} percent",
        text,
    )


def _sub_multipliers(text: str) -> str:
    """"2x" -> "two times"; "40+" -> "over forty"."""
    text = re.sub(
        r"\b(\d[\d,]*(?:\.\d+)?)\s*[xX]\b",
        lambda m: f"{_cardinal(m.group(1))} times",
        text,
    )
    return re.sub(
        r"\b(\d[\d,]*)\+",
        lambda m: f"over {_cardinal(m.group(1))}",
        text,
    )


def _sub_year_ranges(text: str) -> str:
    def repl(m):
        a, b = int(m.group(1)), int(m.group(2))
        if YEAR_MIN <= a <= YEAR_MAX and YEAR_MIN <= b <= YEAR_MAX:
            return f"{_year(a)} to {_year(b)}"
        return m.group(0)

    return re.sub(r"\b(\d{4})\s*[-–]\s*(\d{4})\b", repl, text)


def _sub_years(text: str) -> str:
    def repl(m):
        n = int(m.group(0))
        return _year(n) if YEAR_MIN <= n <= YEAR_MAX else m.group(0)

    return re.sub(r"\b\d{4}\b", repl, text)


def _sub_ordinals(text: str) -> str:
    return re.sub(
        r"\b(\d+)(st|nd|rd|th)\b",
        lambda m: num2words(int(m.group(1)), to="ordinal"),
        text,
        flags=re.I,
    )


def _sub_cardinals(text: str) -> str:
    return re.sub(r"\b\d[\d,]*(?:\.\d+)?\b", lambda m: _cardinal(m.group(0)), text)


def normalize_numbers(text: str) -> str:
    """Order matters: more specific patterns consume their digits first."""
    text = _sub_currency(text)
    text = _sub_percent(text)
    text = _sub_multipliers(text)
    text = _sub_year_ranges(text)
    text = _sub_years(text)
    text = _sub_ordinals(text)
    text = _sub_cardinals(text)
    return text


def normalize_punctuation(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    # Em/en dashes become commas so the model produces a pause rather than
    # running the clauses together.
    text = re.sub(r"\s*[—–]\s*", ", ", text)
    text = re.sub(r"\s+-\s+", ", ", text)
    text = text.replace("…", ",")
    text = re.sub(r"\.{3,}", ",", text)
    text = text.replace("&", " and ")
    # Removing citations leaves gaps like "decisions ." -- close them up.
    text = re.sub(r"[ \t]+([,.!?;:])", r"\1", text)
    # Collapse the comma pileups the substitutions above can create.
    text = re.sub(r"(,\s*){2,}", ", ", text)
    text = re.sub(r",\s*([.!?])", r"\1", text)
    return text


def collapse_whitespace(text: str) -> str:
    """Squash runs of spaces but keep blank lines as paragraph markers."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def normalize(text: str) -> str:
    """Full normalization pass. Run this before chunking."""
    text = _MD_CODE_FENCE.sub(" ", text)
    text = _MD_BLOCKQUOTE.sub("", text)
    text = unwrap_soft_breaks(text)
    text = strip_markdown(text)
    text = _CITATION.sub("", text)
    text = _FOOTNOTE.sub("", text)
    text = expand_abbreviations(text)
    text = normalize_numbers(text)
    text = normalize_punctuation(text)
    return collapse_whitespace(text)
