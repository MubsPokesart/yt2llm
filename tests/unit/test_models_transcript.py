"""Tests for Word, Segment, and Transcript models."""

import pytest
from pydantic import ValidationError

from yt2md.models import Segment, Transcript, Word


class TestWord:
    def test_minimal_word(self) -> None:
        w = Word(text="hello", start=0.0, end=0.5, speaker=None)
        assert w.text == "hello"
        assert w.start == pytest.approx(0.0)
        assert w.end == pytest.approx(0.5)
        assert w.speaker is None

    def test_word_with_speaker(self) -> None:
        w = Word(text="hi", start=1.0, end=1.2, speaker="SPEAKER_00")
        assert w.speaker == "SPEAKER_00"

    def test_end_before_start_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Word(text="bad", start=1.0, end=0.5, speaker=None)

    def test_negative_start_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Word(text="bad", start=-0.1, end=0.5, speaker=None)


class TestSegment:
    def test_segment_with_words(self) -> None:
        words = [
            Word(text="hello", start=0.0, end=0.5, speaker="S0"),
            Word(text="world", start=0.6, end=1.0, speaker="S0"),
        ]
        s = Segment(
            start=0.0,
            end=1.0,
            text="hello world",
            speaker="S0",
            words=words,
        )
        assert len(s.words) == 2
        assert s.text == "hello world"


class TestTranscript:
    def test_transcript_minimal(self) -> None:
        t = Transcript(
            language="en",
            duration_s=1.0,
            backend="openai_transcribe",
            model_id="gpt-4o-transcribe",
            chunked=False,
            segments=[
                Segment(
                    start=0.0,
                    end=1.0,
                    text="hi",
                    speaker=None,
                    words=[Word(text="hi", start=0.0, end=1.0, speaker=None)],
                ),
            ],
            speakers=[],
        )
        assert t.backend == "openai_transcribe"
        assert t.chunked is False
        assert len(t.segments) == 1

    def test_invalid_backend_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Transcript(
                language="en",
                duration_s=1.0,
                backend="not_a_backend",  # type: ignore[arg-type]
                model_id="x",
                chunked=False,
                segments=[],
                speakers=[],
            )

    def test_round_trip_json(self) -> None:
        t = Transcript(
            language="en",
            duration_s=2.5,
            backend="local_whisper",
            model_id="faster-whisper-medium",
            chunked=True,
            segments=[],
            speakers=["Alice"],
        )
        as_json = t.model_dump_json()
        back = Transcript.model_validate_json(as_json)
        assert back == t
