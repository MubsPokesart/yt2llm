"""Tests for filler-word removal in the clean stage."""

import pytest

from yt2md.models import Segment, Transcript, Word
from yt2md.stages.clean import clean


def _word(text: str, start: float, end: float, speaker: str | None = "S0") -> Word:
    return Word(text=text, start=start, end=end, speaker=speaker)


def _segment(words: list[Word], speaker: str | None = "S0") -> Segment:
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(w.text for w in words),
        speaker=speaker,
        words=words,
    )


def _transcript(segments: list[Segment], speakers: list[str]) -> Transcript:
    duration = segments[-1].end if segments else 0.0
    return Transcript(
        language="en",
        duration_s=duration,
        backend="openai_transcribe",
        model_id="gpt-4o-transcribe",
        chunked=False,
        segments=segments,
        speakers=speakers,
    )


class TestTextOnlySegmentFallback:
    """When a transcript backend returns segments with text but no words array
    (e.g., faster-whisper without word_timestamps=True, or any future backend),
    the cleaner must not drop the segment — that erases the entire transcript.
    Filler removal falls back to filtering tokens from the text directly.
    """

    def test_text_only_segment_preserved(self) -> None:
        text_only = Segment(
            start=0.0,
            end=2.0,
            text="hello world",
            speaker="S0",
            words=[],
        )
        t = Transcript(
            language="en",
            duration_s=2.0,
            backend="openai_transcribe",
            model_id="m",
            chunked=False,
            segments=[text_only],
            speakers=[],
        )
        result = clean(t)
        assert len(result.segments) == 1
        assert result.segments[0].text == "hello world"

    def test_text_only_segment_drops_fillers_from_text(self) -> None:
        text_only = Segment(
            start=0.0,
            end=2.0,
            text="uh hello um world",
            speaker="S0",
            words=[],
        )
        t = Transcript(
            language="en",
            duration_s=2.0,
            backend="openai_transcribe",
            model_id="m",
            chunked=False,
            segments=[text_only],
            speakers=[],
        )
        result = clean(t)
        assert len(result.segments) == 1
        assert result.segments[0].text == "hello world"


class TestFillerRemoval:
    def test_uh_dropped(self) -> None:
        t = _transcript(
            [_segment([_word("Uh", 0.0, 0.2), _word("hello", 0.3, 1.0)])],
            ["S0"],
        )
        result = clean(t)
        words = result.segments[0].words
        assert [w.text for w in words] == ["hello"]

    def test_uh_with_comma_dropped(self) -> None:
        t = _transcript(
            [_segment([_word("Uh,", 0.0, 0.2), _word("hello", 0.3, 1.0)])],
            ["S0"],
        )
        result = clean(t)
        words = result.segments[0].words
        assert [w.text for w in words] == ["hello"]

    def test_um_dropped(self) -> None:
        t = _transcript(
            [_segment([_word("um", 0.0, 0.2), _word("yes", 0.3, 1.0)])],
            ["S0"],
        )
        result = clean(t)
        assert [w.text for w in result.segments[0].words] == ["yes"]

    def test_all_hard_fillers_dropped(self) -> None:
        words = [
            _word("uh", 0.0, 0.1),
            _word("um", 0.2, 0.3),
            _word("er", 0.4, 0.5),
            _word("ah", 0.6, 0.7),
            _word("uhm", 0.8, 0.9),
            _word("hello", 1.0, 1.5),
        ]
        t = _transcript([_segment(words)], ["S0"])
        result = clean(t)
        assert [w.text for w in result.segments[0].words] == ["hello"]

    def test_like_preserved(self) -> None:
        # "like" is intentionally NOT a hard filler.
        t = _transcript(
            [_segment([_word("like", 0.0, 0.3), _word("water", 0.4, 1.0)])],
            ["S0"],
        )
        result = clean(t)
        assert [w.text for w in result.segments[0].words] == ["like", "water"]

    def test_mm_preserved(self) -> None:
        # "mm" is an agreement sound, not a filler.
        t = _transcript([_segment([_word("Mm.", 0.0, 0.5)])], ["S0"])
        result = clean(t)
        assert len(result.segments) == 1
        assert result.segments[0].words[0].text == "Mm."

    def test_surviving_word_timestamps_unchanged(self) -> None:
        t = _transcript(
            [
                _segment([
                    _word("Uh", 0.0, 0.2),
                    _word("dopamine", 0.5, 1.5),
                    _word("signals", 1.6, 2.4),
                ])
            ],
            ["S0"],
        )
        result = clean(t)
        words = result.segments[0].words
        assert words[0].text == "dopamine"
        assert words[0].start == pytest.approx(0.5)
        assert words[0].end == pytest.approx(1.5)
        assert words[1].start == pytest.approx(1.6)
        assert words[1].end == pytest.approx(2.4)

    def test_segment_text_rebuilt_from_surviving_words(self) -> None:
        t = _transcript(
            [
                _segment([
                    _word("Uh,", 0.0, 0.2),
                    _word("dopamine", 0.5, 1.5),
                    _word("signals", 1.6, 2.4),
                ])
            ],
            ["S0"],
        )
        result = clean(t)
        assert result.segments[0].text == "dopamine signals"
