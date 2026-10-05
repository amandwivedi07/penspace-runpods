"""The Cartesia backend, tested without spending a credit.

Every HTTP call is stubbed. What is worth pinning here is not that urllib
works — it is the handful of behaviours that cost money or silently produce
the wrong audio when they regress.

    pip install pytest && pytest penspace/tests -q
"""

import io
import json
import urllib.error

import numpy as np
import pytest

from penspace.cartesia import CartesiaSynthesizer, CartesiaError, cartesia_language
from penspace.config import Config
from penspace.runner import Runner, build_synthesizer


def cfg_for(**over) -> Config:
    cfg = Config()
    cfg.backend = "cartesia"
    cfg.cartesia_api_key = "k"
    cfg.cartesia_voice_id = "voice-1"
    cfg.cartesia_max_retries = 2
    cfg.cartesia_backoff = 0.0
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def pcm(n: int = 8) -> bytes:
    return np.full(n, 0.25, dtype=np.float32).tobytes()


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class TestRenderIdentity:
    def test_switching_backend_changes_the_render_id(self):
        """The S3 skip is keyed on this.

        Qwen and Sonic narrating the same chapter are two different
        recordings. If the fingerprint could not tell them apart, switching
        backends would hand back the OLD audio for every chapter already
        rendered — the worst kind of bug, because it looks like it worked.
        """
        assert Config().active_model_path != cfg_for().active_model_path

    def test_switching_voice_changes_the_render_id(self):
        a = cfg_for(cartesia_voice_id="julian")
        b = cfg_for(cartesia_voice_id="andre")
        assert a.active_model_path != b.active_model_path

    def test_the_factory_honours_the_backend(self):
        assert isinstance(build_synthesizer(cfg_for()), CartesiaSynthesizer)
        assert not isinstance(build_synthesizer(Config()), CartesiaSynthesizer)


class TestSpend:
    def test_only_successful_requests_are_billed(self, monkeypatch):
        """A request that never returned audio is not on the invoice."""
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError(
                    req.full_url, 500, "boom", {}, io.BytesIO(b"server error")
                )
            return FakeResponse(pcm())

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        synth = CartesiaSynthesizer(cfg_for())
        synth._api_key = "k"

        out = synth.synthesize(["hello"], [0])
        assert len(out) == 1
        # Two requests, one success, five characters billed — not ten.
        assert calls["n"] == 2
        assert synth.characters_sent == 5

    def test_credits_are_characters(self):
        assert CartesiaSynthesizer.estimate_credits(["abc", "de"]) == 5


class TestRequest:
    def _capture(self, monkeypatch):
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            seen["headers"] = {k.lower(): v for k, v in req.headers.items()}
            seen["body"] = json.loads(req.data.decode())
            return FakeResponse(pcm())

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        return seen

    def test_the_job_language_wins_over_the_endpoint_default(self, monkeypatch):
        """Same lesson as the local backend: the manifest's language, not the
        machine's. German narrated as English is the bug this prevents."""
        seen = self._capture(monkeypatch)
        synth = CartesiaSynthesizer(cfg_for(language="English"))
        synth._api_key = "k"
        synth.synthesize(["Guten Tag"], [0], language="German")
        assert seen["body"]["language"] == "de"

    def test_raw_float_samples_are_requested(self, monkeypatch):
        """Raw f32 needs no decoder and is what assemble() already expects."""
        seen = self._capture(monkeypatch)
        synth = CartesiaSynthesizer(cfg_for())
        synth._api_key = "k"
        synth.synthesize(["hi"], [0])
        assert seen["body"]["output_format"]["encoding"] == "pcm_f32le"
        assert seen["body"]["output_format"]["container"] == "raw"
        assert seen["headers"]["cartesia-version"]

    def test_chunks_keep_their_index_when_they_finish_out_of_order(self, monkeypatch):
        """Requests run concurrently, so completion order is not input order.
        The caller keys results by index; if that drifted, chapters would be
        assembled with their paragraphs shuffled."""
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda req, timeout=None: FakeResponse(pcm())
        )
        synth = CartesiaSynthesizer(cfg_for())
        synth._api_key = "k"
        out = synth.synthesize(["a", "b", "c"], [7, 8, 9])
        assert sorted(c.index for c in out) == [7, 8, 9]
        assert all(c.sample_rate == 44100 and not c.truncated for c in out)


class TestFailures:
    def test_a_client_error_is_not_retried(self, monkeypatch):
        """A 400 means the request is wrong. Retrying it three times just
        spends three times as long being wrong."""
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            raise urllib.error.HTTPError(
                req.full_url, 400, "bad", {}, io.BytesIO(b"bad transcript")
            )

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        synth = CartesiaSynthesizer(cfg_for())
        synth._api_key = "k"
        with pytest.raises(CartesiaError, match="400"):
            synth.synthesize(["x"], [0])
        assert calls["n"] == 1

    def test_a_missing_key_fails_before_any_request(self):
        synth = CartesiaSynthesizer(cfg_for(cartesia_api_key=None))
        import os

        if os.environ.get("CARTESIA_API_KEY"):
            pytest.skip("a real key is set in this environment")
        with pytest.raises(CartesiaError, match="API key"):
            synth.load()


class TestLanguage:
    @pytest.mark.parametrize(
        "given,expected",
        [("German", "de"), ("de", "de"), ("Japanese", "ja"), ("Hindi", "hi"),
         ("en-GB", "en-gb"), (None, "en")],
    )
    def test_names_and_codes_both_resolve(self, given, expected):
        assert cartesia_language(given) == expected
