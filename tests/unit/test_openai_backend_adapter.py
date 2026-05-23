"""Tests for the OpenAI response → Transcript adapter (no network)."""

import json
from pathlib import Path

import pytest

from yt2md.models import Transcript
from yt2md.stages.transcribe_backends.openai import normalize_openai_response


@pytest.fixture
def raw(fixtures_dir: Path) -> dict[str, object]:
    return json.loads(
        (fixtures_dir / "transcripts" / "openai_raw_sample.json").read_text(encoding="utf-8")
    )


class TestNormalizeOpenAIResponse:
    def test_returns_transcript(self, raw: dict[str, object]) -> None:
        t = normalize_openai_response(raw, model_id="gpt-4o-transcribe")
        assert isinstance(t, Transcript)

    def test_backend_field(self, raw: dict[str, object]) -> None:
        t = normalize_openai_response(raw, model_id="gpt-4o-transcribe")
        assert t.backend == "openai_transcribe"
        assert t.model_id == "gpt-4o-transcribe"

    def test_duration_mapped(self, raw: dict[str, object]) -> None:
        t = normalize_openai_response(raw, model_id="gpt-4o-transcribe")
        assert t.duration_s == pytest.approx(8.0)

    def test_speakers_collected(self, raw: dict[str, object]) -> None:
        t = normalize_openai_response(raw, model_id="gpt-4o-transcribe")
        assert t.speakers == ["SPEAKER_00"]

    def test_word_timestamps_preserved(self, raw: dict[str, object]) -> None:
        t = normalize_openai_response(raw, model_id="gpt-4o-transcribe")
        first_word = t.segments[0].words[0]
        assert first_word.text == "Hello"  # leading space stripped
        assert first_word.start == pytest.approx(0.0)
        assert first_word.end == pytest.approx(0.5)
        assert first_word.speaker == "SPEAKER_00"

    def test_chunked_flag_false_by_default(self, raw: dict[str, object]) -> None:
        t = normalize_openai_response(raw, model_id="gpt-4o-transcribe")
        assert t.chunked is False

    def test_undiarized_response_yields_none_speakers(self) -> None:
        no_speaker = {
            "language": "en",
            "duration": 2.0,
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.0,
                    "text": " Hi.",
                    "words": [{"word": " Hi.", "start": 0.0, "end": 2.0}],
                }
            ],
            "text": "Hi.",
        }
        t = normalize_openai_response(no_speaker, model_id="gpt-4o-transcribe")
        assert t.speakers == []
        assert t.segments[0].speaker is None
        assert t.segments[0].words[0].speaker is None
