"""Tests for structure() — Gemini call mocked + validation retry semantics."""

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from yt2md.config import Config
from yt2md.errors import InvalidStructuredOutputError
from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    StructuredDoc,
    Transcript,
    VideoMetadata,
)
from yt2md.stages.structure import (
    _call_gemini_inner,  # noqa: PLC2701  -- testing retry decorator behavior
    structure,
)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        google_api_key="g",  # type: ignore[arg-type]
        cache_dir=tmp_path,
    )


def _meta() -> VideoMetadata:
    return VideoMetadata(
        video_id="vid",
        url="u",
        title="T",
        channel="C",
        channel_id="UC",
        published_date=date(2025, 1, 1),
        duration_s=60.0,
        description="",
        chapters=[],
        tags=[],
        language=None,
    )


def _transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_s=60.0,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=[],
        speakers=[],
    )


def _valid_response_json() -> str:
    return json.dumps({
        "frontmatter": {
            "title": "T",
            "channel": "C",
            "url": "u",
            "video_id": "vid",
            "published": "2025-01-01",
            "duration_seconds": 60,
            "captured_at": "2026-05-23",
            "schema_version": CURRENT_SCHEMA_VERSION,
            "genre": "podcast",
            "speakers": ["A"],
            "topics": [],
            "people_mentioned": [],
            "works_mentioned": [],
        },
        "tldr": "Hello.",
        "takeaways": [
            {"text": "t1", "timestamp_s": 0.0},
            {"text": "t2", "timestamp_s": 1.0},
            {"text": "t3", "timestamp_s": 2.0},
        ],
        "concepts": [],
        "references": [],
        "quotes": [],
        "sections": [],
        "open_questions": [],
        "speaker_mappings": [],
    })


def _invalid_response_json() -> str:
    # Only 1 takeaway → validation fails (need ≥3)
    data = json.loads(_valid_response_json())
    data["takeaways"] = [{"text": "only-one", "timestamp_s": 0.0}]
    return json.dumps(data)


def _fake_gemini_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


class TestStructureHappy:
    def test_returns_structured_doc(self, cfg: Config) -> None:
        with patch("yt2md.stages.structure._call_gemini") as call:
            call.return_value = _fake_gemini_response(_valid_response_json())
            doc = structure(_transcript(), _meta(), cfg=cfg)
        assert isinstance(doc, StructuredDoc)
        assert len(doc.takeaways) == 3


class TestGeminiUsesResponseJsonSchema:
    """We pass the full Pydantic schema via `response_json_schema` (not `response_schema`).

    The SDK validates `response_schema` client-side via `_raise_for_unsupported_mldev_properties`,
    rejecting `additionalProperties` on the Developer API even though the server accepts it
    (filed upstream as googleapis/python-genai#1815). The `response_json_schema` field
    bypasses that validator. Using it lets us hand the raw Pydantic schema in without
    pre-processing.
    """

    def test_call_uses_response_json_schema_not_response_schema(self, cfg: Config) -> None:
        fake_models = MagicMock()
        fake_models.generate_content.return_value = _fake_gemini_response(_valid_response_json())
        client = MagicMock()
        client.models = fake_models
        with patch("yt2md.stages.structure.genai.Client", return_value=client):
            _call_gemini_inner("prompt", cfg)
        config = fake_models.generate_content.call_args.kwargs["config"]
        assert config.response_json_schema is not None
        assert config.response_schema is None


class TestGeminiTransientRetry:
    """Tenacity retry on _call_gemini_inner must cover the SDK's own transient errors.

    The OpenAI backend retries on RateLimitError / InternalServerError. The Gemini
    backend must mirror that: a 429 ClientError or 5xx ServerError should back off
    and retry, not fail the first attempt. Compare openai.py:106-111.
    """

    def _patched_client(self, side_effect: list[object]) -> MagicMock:
        fake_models = MagicMock()
        fake_models.generate_content.side_effect = side_effect
        client = MagicMock()
        client.models = fake_models
        return client

    def test_retries_on_server_error_then_succeeds(self, cfg: Config) -> None:
        responses = [
            genai_errors.ServerError(503, {"error": {"message": "unavailable"}}),
            _fake_gemini_response(_valid_response_json()),
        ]
        client = self._patched_client(responses)
        with patch("yt2md.stages.structure.genai.Client", return_value=client):
            result = _call_gemini_inner("prompt", cfg)
        assert client.models.generate_content.call_count == 2
        assert result.text == _valid_response_json()

    def test_retries_on_rate_limit_client_error(self, cfg: Config) -> None:
        responses = [
            genai_errors.ClientError(429, {"error": {"message": "rate limited"}}),
            _fake_gemini_response(_valid_response_json()),
        ]
        client = self._patched_client(responses)
        with patch("yt2md.stages.structure.genai.Client", return_value=client):
            result = _call_gemini_inner("prompt", cfg)
        assert client.models.generate_content.call_count == 2
        assert result.text == _valid_response_json()

    def test_does_not_retry_on_4xx_other_than_429(self, cfg: Config) -> None:
        """A 400 invalid-request should fail fast — retrying won't fix bad input."""
        responses = [
            genai_errors.ClientError(400, {"error": {"message": "bad request"}}),
        ]
        client = self._patched_client(responses)
        with (
            patch("yt2md.stages.structure.genai.Client", return_value=client),
            pytest.raises(genai_errors.ClientError),
        ):
            _call_gemini_inner("prompt", cfg)
        assert client.models.generate_content.call_count == 1


class TestGeminiCallConfigDeterminism:
    """Gemini call must NOT set seed from Python's built-in hash().

    Python's hash() is salted per interpreter (PYTHONHASHSEED). A seed derived from
    hash(prompt) is non-deterministic across processes — undermining the
    determinism it appears to provide. Drop it; temperature=0.2 already pins variance.
    """

    def test_no_python_hash_seed_in_generate_content_config(self, cfg: Config) -> None:
        fake_models = MagicMock()
        fake_models.generate_content.return_value = _fake_gemini_response(_valid_response_json())
        client = MagicMock()
        client.models = fake_models
        with patch("yt2md.stages.structure.genai.Client", return_value=client):
            _call_gemini_inner("prompt-text", cfg)
        kwargs = fake_models.generate_content.call_args.kwargs
        config = kwargs["config"]
        # Either no seed attribute, or seed is None — both mean "not using hash()".
        assert getattr(config, "seed", None) is None, (
            f"Gemini call should not set seed (Python hash() is per-process salted); "
            f"got seed={config.seed}"
        )


class TestStructureRetryOnValidation:
    def test_retries_once_on_invalid_then_succeeds(self, cfg: Config) -> None:
        responses = [
            _fake_gemini_response(_invalid_response_json()),
            _fake_gemini_response(_valid_response_json()),
        ]
        with patch("yt2md.stages.structure._call_gemini", side_effect=responses) as call:
            doc = structure(_transcript(), _meta(), cfg=cfg)
        assert call.call_count == 2
        assert isinstance(doc, StructuredDoc)

    def test_raises_after_second_failure(self, cfg: Config) -> None:
        with patch("yt2md.stages.structure._call_gemini") as call:
            call.return_value = _fake_gemini_response(_invalid_response_json())
            with pytest.raises(InvalidStructuredOutputError):
                structure(_transcript(), _meta(), cfg=cfg)
            assert call.call_count == 2
