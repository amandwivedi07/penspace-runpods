"""Regression tests for normalization and chunking.

Every case here is a bug that shipped silently at some point: text that
disappeared, words that fused together, or paragraphs that merged. They are all
invisible until you listen to the audio, which is exactly why they are pinned
here.

    pip install pytest && pytest penspace/tests -q
"""

import pytest

from penspace.chunker import chunk_text, split_sentences
from penspace.config import Config
from penspace.normalize import normalize
from penspace.runner import Runner, _reference_fingerprint, prepare_clone


class TestNumberDeletion:
    def test_wrapped_number_is_not_eaten_as_a_list_marker(self):
        # "35." landed at the start of a soft-wrapped line and looked exactly
        # like an ordered-list bullet, so it was deleted outright.
        text = (
            "people estimate a far higher age than when the anchor is\n"
            "35. Availability: we judge frequency by ease of recall."
        )
        assert "thirty-five" in normalize(text)

    def test_real_numbered_list_still_loses_its_markers(self):
        text = "The four laws:\n\n1. Make it obvious.\n2. Make it easy."
        out = normalize(text)
        assert "Make it obvious." in out
        assert "one. Make" not in out

    def test_wrapped_list_item_rejoins(self):
        text = "Laws:\n\n1. Make it attractive, especially when\nthe cue is ambiguous."
        assert "attractive, especially when the cue is ambiguous" in normalize(text)


class TestCurrency:
    def test_space_after_amount_survives(self):
        # The optional scale-word group used to swallow the trailing space,
        # producing "one hundred and fifty dollarsor".
        assert "dollars or" in normalize("A coin flip to win $150 or lose $200.")

    def test_singular_and_plural(self):
        assert "one dollar" in normalize("It costs $1.")
        assert "one hundred dollars" in normalize("It costs $100.")

    def test_scale_words(self):
        assert "five million dollars" in normalize("He raised $5 million.")


class TestNumberForms:
    def test_years(self):
        assert "twenty eleven" in normalize("Published in 2011.")

    def test_year_ranges(self):
        assert "nineteen sixty-nine to nineteen seventy-nine" in normalize(
            "Work from 1969-1979."
        )

    def test_multiplier(self):
        assert "two times" in normalize("Losses hurt 2x as much.")

    def test_plus_suffix(self):
        assert "over forty years" in normalize("distills 40+ years of research")

    def test_percent(self):
        assert "ninety-eight percent" in normalize("Roughly 98% of judgments.")

    def test_ordinal(self):
        assert "twenty-first" in normalize("The 21st century.")


class TestParagraphs:
    def test_blockquote_stays_its_own_paragraph(self):
        # `^\s{0,3}>` matched the newline before the quote and deleted the blank
        # line, merging the quote into the preceding paragraph.
        text = "People overestimate plane crashes.\n\n> Nothing in life is as\n> important.\n\nProspect theory followed."
        chunks = chunk_text(normalize(text), 300)
        assert len(chunks) == 3
        assert chunks[1].text == "Nothing in life is as important."

    def test_soft_wraps_are_joined(self):
        chunks = chunk_text(normalize("One sentence that\nwraps across lines."), 300)
        assert chunks[0].text == "One sentence that wraps across lines."

    def test_paragraph_ends_are_flagged(self):
        text = "First para sentence one. Sentence two.\n\nSecond para here."
        chunks = chunk_text(normalize(text), 40)
        assert chunks[-1].ends_paragraph
        assert sum(1 for c in chunks if c.ends_paragraph) == 2


class TestMarkdown:
    def test_citations_leave_no_gap(self):
        assert "decisions." in normalize("how the mind makes decisions [1].")

    def test_links_keep_their_label(self):
        assert normalize("See [the study](http://x.com/y).") == "See the study."

    def test_emphasis_is_stripped(self):
        assert normalize("**Thinking** and *Slow*") == "Thinking and Slow"


class TestChunking:
    def test_never_exceeds_limit_on_ordinary_prose(self):
        text = " ".join(f"This is sentence number {i} in a long run." for i in range(60))
        assert all(len(c.text) <= 200 for c in chunk_text(normalize(text), 200))

    def test_long_sentence_splits_at_clauses(self):
        text = "alpha " * 30 + ", " + "beta " * 30 + ", " + "gamma " * 30 + "."
        assert len(chunk_text(normalize(text), 200)) > 1

    def test_initials_do_not_end_a_sentence(self):
        assert len(split_sentences("Work by J. K. Rowling changed things.")) == 1

    def test_indices_are_contiguous(self):
        text = "\n\n".join(f"Paragraph {i} has a sentence." for i in range(5))
        chunks = chunk_text(normalize(text), 100)
        assert [c.index for c in chunks] == list(range(len(chunks)))


class TestCloneConfig:
    def test_mode_selects_the_checkpoint(self):
        cfg = Config()
        assert not cfg.is_clone
        assert "CustomVoice" in cfg.active_model_path

        cfg.voice_mode = "clone"
        assert cfg.is_clone
        # Cloning requires the Base checkpoint, not CustomVoice.
        assert cfg.active_model_path.endswith("Base")

    def test_reference_bytes_drive_the_fingerprint(self, tmp_path):
        a, b = tmp_path / "a.wav", tmp_path / "b.wav"
        a.write_bytes(b"reference-one")
        b.write_bytes(b"reference-two")

        cfg = Config(voice_mode="clone", ref_audio=str(a), ref_text="hello")
        first = _reference_fingerprint(cfg)
        cfg.ref_audio = str(b)
        assert _reference_fingerprint(cfg) != first

    def test_transcript_change_changes_the_fingerprint(self, tmp_path):
        ref = tmp_path / "a.wav"
        ref.write_bytes(b"reference")
        cfg = Config(voice_mode="clone", ref_audio=str(ref), ref_text="hello")
        first = _reference_fingerprint(cfg)
        cfg.ref_text = "different transcript"
        assert _reference_fingerprint(cfg) != first

    def test_render_id_differs_between_voices(self, tmp_path):
        ref = tmp_path / "voice.wav"
        ref.write_bytes(b"reference")
        text = "Small habits compound."

        # `language` became required when the renderer learned to take it from
        # the job rather than the endpoint; hold it fixed so this test keeps
        # measuring the voice.
        builtin = Runner(Config(), tmp_path)._render_id(text, "English")
        cloned = Runner(
            Config(voice_mode="clone", ref_audio=str(ref), ref_text="hi"), tmp_path
        )._render_id(text, "English")

        # Same text, different voice -> different S3 key, so a re-render never
        # serves stale audio from the previous voice.
        assert builtin != cloned

    def test_clone_without_reference_is_rejected(self):
        with pytest.raises(ValueError, match="ref-audio"):
            prepare_clone(Config(voice_mode="clone"))

    def test_custom_voice_mode_is_untouched(self):
        cfg = Config()
        assert prepare_clone(cfg) is cfg
        assert cfg.ref_text is None
