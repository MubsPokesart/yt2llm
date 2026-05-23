"""Tests for chunk.stitch_transcripts() — combine per-chunk transcripts with offset timestamps."""

import pytest

from yt2md.models import Segment, Transcript, Word
from yt2md.stages.chunk import stitch_transcripts


def _t(start: float, end: float, segments: list[Segment]) -> Transcript:
    return Transcript(
        language="en",
        duration_s=end - start,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=segments,
        speakers=[],
    )


def _seg(start: float, end: float, text: str) -> Segment:
    return Segment(
        start=start,
        end=end,
        text=text,
        speaker=None,
        words=[Word(text=text, start=start, end=end, speaker=None)],
    )


class TestStitch:
    def test_single_chunk_passthrough(self) -> None:
        t = _t(0.0, 10.0, [_seg(0.0, 10.0, "hi")])
        stitched = stitch_transcripts([t], offsets_s=[0.0])
        assert stitched == t.model_copy(update={"chunked": True})

    def test_two_chunks_offset_applied(self) -> None:
        c1 = _t(0.0, 30.0, [_seg(0.0, 10.0, "a"), _seg(10.0, 30.0, "b")])
        c2 = _t(0.0, 20.0, [_seg(0.0, 20.0, "c")])
        stitched = stitch_transcripts([c1, c2], offsets_s=[0.0, 30.0])

        starts = [s.start for s in stitched.segments]
        assert starts == [pytest.approx(0.0), pytest.approx(10.0), pytest.approx(30.0)]

        ends = [s.end for s in stitched.segments]
        assert ends == [pytest.approx(10.0), pytest.approx(30.0), pytest.approx(50.0)]

    def test_word_timestamps_also_offset(self) -> None:
        c1 = _t(0.0, 5.0, [_seg(0.0, 5.0, "a")])
        c2 = _t(0.0, 5.0, [_seg(0.0, 5.0, "b")])
        stitched = stitch_transcripts([c1, c2], offsets_s=[0.0, 5.0])
        c2_word = stitched.segments[1].words[0]
        assert c2_word.start == pytest.approx(5.0)
        assert c2_word.end == pytest.approx(10.0)

    def test_chunked_flag_true(self) -> None:
        t = _t(0.0, 1.0, [_seg(0.0, 1.0, "x")])
        stitched = stitch_transcripts([t], offsets_s=[0.0])
        assert stitched.chunked is True

    def test_total_duration_is_max_end(self) -> None:
        c1 = _t(0.0, 30.0, [_seg(0.0, 30.0, "a")])
        c2 = _t(0.0, 20.0, [_seg(0.0, 20.0, "b")])
        stitched = stitch_transcripts([c1, c2], offsets_s=[0.0, 30.0])
        assert stitched.duration_s == pytest.approx(50.0)
