"""Tests for transcription cost calculators."""

import pytest

from yt2md.costs import local_whisper_cost, openai_transcribe_cost


class TestOpenAITranscribe:
    def test_one_minute(self) -> None:
        # $0.006 per minute as of mid-2025 (codified; bump on price changes)
        cost = openai_transcribe_cost(duration_s=60.0)
        assert cost == pytest.approx(0.006, rel=1e-3)

    def test_zero_duration(self) -> None:
        assert openai_transcribe_cost(duration_s=0.0) == pytest.approx(0.0)

    def test_two_hours(self) -> None:
        cost = openai_transcribe_cost(duration_s=2 * 3600.0)
        assert cost == pytest.approx(0.72, rel=1e-3)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            openai_transcribe_cost(duration_s=-1.0)


class TestLocalWhisper:
    def test_is_zero(self) -> None:
        assert local_whisper_cost(duration_s=3600.0) == pytest.approx(0.0)

    def test_zero_for_zero(self) -> None:
        assert local_whisper_cost(duration_s=0.0) == pytest.approx(0.0)
