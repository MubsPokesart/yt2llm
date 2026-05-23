"""Tests for build_structure_prompt() — transcript with inline timestamps."""

from yt2md.models import Segment, Transcript, VideoMetadata, Word
from yt2md.stages.structure import build_structure_prompt


def _t(*, segments: list[Segment]) -> Transcript:
    duration = max((s.end for s in segments), default=0.0)
    return Transcript(
        language="en",
        duration_s=duration,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=segments,
        speakers=["SPEAKER_00"],
    )


def _seg(start: float, end: float, text: str, speaker: str | None = "SPEAKER_00") -> Segment:
    return Segment(
        start=start,
        end=end,
        text=text,
        speaker=speaker,
        words=[Word(text=text, start=start, end=end, speaker=speaker)],
    )


class TestPromptStructure:
    def test_contains_metadata_section(self, huberman_metadata: VideoMetadata) -> None:
        t = _t(segments=[_seg(0.0, 5.0, "hello")])
        prompt = build_structure_prompt(t, huberman_metadata)
        assert "title" in prompt.lower()
        assert "Huberman Lab" in prompt

    def test_contains_transcript_section(self, huberman_metadata: VideoMetadata) -> None:
        t = _t(segments=[_seg(252.0, 260.0, "Dopamine signals anticipation.")])
        prompt = build_structure_prompt(t, huberman_metadata)
        # Inline [mm:ss] marker for 252s = 04:12
        assert "[04:12]" in prompt
        assert "Dopamine signals anticipation." in prompt

    def test_chunked_flag_propagated(self, huberman_metadata: VideoMetadata) -> None:
        t = _t(segments=[_seg(0.0, 5.0, "x")])
        t_chunked = t.model_copy(update={"chunked": True})
        prompt = build_structure_prompt(t_chunked, huberman_metadata)
        # The prompt should warn the model about speaker labels being unreliable across chunks
        assert "chunk" in prompt.lower()
