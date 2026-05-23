"""Tests for the 95% speaker-collapse rule in the clean stage."""

from yt2md.models import Segment, Transcript, Word
from yt2md.stages.clean import clean


def _word(text: str, start: float, end: float, speaker: str) -> Word:
    return Word(text=text, start=start, end=end, speaker=speaker)


def _seg(start: float, end: float, speaker: str) -> Segment:
    word = _word("x", start, end, speaker)
    return Segment(start=start, end=end, text="x", speaker=speaker, words=[word])


class TestSpeakerCollapse:
    def test_collapse_at_96_percent(self) -> None:
        # SPEAKER_00 = 96s, SPEAKER_01 = 4s -> 0.96 >= 0.95 -> collapse.
        t = Transcript(
            language="en",
            duration_s=100.0,
            backend="openai_transcribe",
            model_id="gpt-4o-transcribe",
            chunked=False,
            segments=[
                _seg(0.0, 96.0, "SPEAKER_00"),
                _seg(96.0, 100.0, "SPEAKER_01"),
            ],
            speakers=["SPEAKER_00", "SPEAKER_01"],
        )
        result = clean(t)
        # Below 1% rule: SPEAKER_01 has 4% so survives the noise filter,
        # but the 95% collapse rule rewrites all to dominant.
        assert result.speakers == ["SPEAKER_00"]
        assert all(s.speaker == "SPEAKER_00" for s in result.segments)
        assert all(w.speaker == "SPEAKER_00" for s in result.segments for w in s.words)

    def test_no_collapse_at_94_percent(self) -> None:
        # SPEAKER_00 = 94s, SPEAKER_01 = 6s -> 0.94 < 0.95 -> no collapse.
        t = Transcript(
            language="en",
            duration_s=100.0,
            backend="openai_transcribe",
            model_id="gpt-4o-transcribe",
            chunked=False,
            segments=[
                _seg(0.0, 94.0, "SPEAKER_00"),
                _seg(94.0, 100.0, "SPEAKER_01"),
            ],
            speakers=["SPEAKER_00", "SPEAKER_01"],
        )
        result = clean(t)
        assert set(result.speakers) == {"SPEAKER_00", "SPEAKER_01"}

    def test_collapse_at_exactly_95_percent(self) -> None:
        # SPEAKER_00 = 95s, SPEAKER_01 = 5s -> 0.95 >= 0.95 -> collapse.
        t = Transcript(
            language="en",
            duration_s=100.0,
            backend="openai_transcribe",
            model_id="gpt-4o-transcribe",
            chunked=False,
            segments=[
                _seg(0.0, 95.0, "SPEAKER_00"),
                _seg(95.0, 100.0, "SPEAKER_01"),
            ],
            speakers=["SPEAKER_00", "SPEAKER_01"],
        )
        result = clean(t)
        assert result.speakers == ["SPEAKER_00"]

    def test_noise_speaker_dropped(self) -> None:
        # SPEAKER_00 = 99.5s, SPEAKER_01 = 0.5s -> 0.5% -> noise, drop segment.
        t = Transcript(
            language="en",
            duration_s=100.0,
            backend="openai_transcribe",
            model_id="gpt-4o-transcribe",
            chunked=False,
            segments=[
                _seg(0.0, 99.5, "SPEAKER_00"),
                _seg(99.5, 100.0, "SPEAKER_01"),
            ],
            speakers=["SPEAKER_00", "SPEAKER_01"],
        )
        result = clean(t)
        assert result.speakers == ["SPEAKER_00"]
        # SPEAKER_01 contributed <1% -> segment dropped, not relabeled.
        assert len(result.segments) == 1
        assert result.segments[0].speaker == "SPEAKER_00"

    def test_undiarized_unchanged(self) -> None:
        # No speakers -> no collapse logic applies.
        t = Transcript(
            language="en",
            duration_s=10.0,
            backend="local_whisper",
            model_id="faster-whisper-medium",
            chunked=False,
            segments=[
                Segment(
                    start=0.0,
                    end=10.0,
                    text="hi",
                    speaker=None,
                    words=[Word(text="hi", start=0.0, end=10.0, speaker=None)],
                ),
            ],
            speakers=[],
        )
        result = clean(t)
        assert result.speakers == []
        assert result.segments[0].speaker is None
