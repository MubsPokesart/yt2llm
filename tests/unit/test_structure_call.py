"""Tests for structure() — Gemini call mocked + validation retry semantics."""

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from google.genai import _transformers  # noqa: PLC2701  -- SDK's private validator

from yt2md.config import Config
from yt2md.errors import InvalidStructuredOutputError
from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    StructuredDoc,
    Transcript,
    VideoMetadata,
)
from yt2md.stages.structure import (
    _build_gemini_schema,  # noqa: PLC2701  -- testing private boundary helper
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
        "speaker_name_map": {},
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


class TestGeminiDeveloperApiSchemaCompat:
    """The schema we hand to Gemini's Developer API must avoid keywords like
    `additionalProperties` that only Vertex/Enterprise mode supports.

    Regression: StructuredDoc.speaker_name_map (dict[str, str]) caused Pydantic
    to emit `additionalProperties` in the JSON Schema. The Gemini SDK validates
    this client-side and raises ValueError before any network call.
    """

    def test_response_schema_excludes_additional_properties_anywhere(self) -> None:
        """Walk the schema; assert no `additionalProperties` at any nesting depth.

        Catches: any future dict[K, V] field added to StructuredDoc or its nested
        models whose `additionalProperties` slips past the boundary helper.
        """
        schema = _build_gemini_schema()
        offenders: list[str] = []

        def walk(node: object, path: str) -> None:
            if isinstance(node, dict):
                if "additionalProperties" in node or "additional_properties" in node:
                    offenders.append(path or "$")
                for k, v in node.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")

        walk(schema, "")
        assert not offenders, (
            f"Gemini Developer API rejects `additionalProperties`; found at: {offenders}"
        )

    def test_response_schema_passes_sdk_developer_api_check(self) -> None:
        """Run the Gemini SDK's own validator on our schema.

        Forward-safe: if the SDK rejects more keywords in a future release,
        this test breaks the moment we upgrade, telling us exactly what to strip.
        """
        stub_client = SimpleNamespace(vertexai=False)
        _transformers.process_schema(_build_gemini_schema(), stub_client)


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
