"""Tests for the Full Cleaned Transcript section.

Requirements:
  - Sections labeled ## Full Cleaned Transcript
  - Words grouped into ~60s paragraphs with [mm:ss] timestamp markers at block start
  - Speaker change forces a new paragraph (even if under 60s)
  - SPEAKER_NN labels substituted using doc.speaker_mappings
"""

from datetime import date

from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    Frontmatter,
    Segment,
    SpeakerMapping,
    StructuredDoc,
    Transcript,
    Word,
)
from yt2md.stages.render import render


def _doc(mappings: dict[str, str]) -> StructuredDoc:
    speaker_mappings = [
        SpeakerMapping(label=label, display_name=name) for label, name in mappings.items()
    ]
    fm = Frontmatter(
        title="T",
        channel="C",
        url="https://www.youtube.com/watch?v=v",
        video_id="v",
        published=date(2025, 1, 1),
        duration_seconds=200,
        captured_at=date(2026, 5, 23),
        schema_version=CURRENT_SCHEMA_VERSION,
        genre="podcast",
        speakers=list(mappings.values()) or ["A"],
        topics=[],
        people_mentioned=[],
        works_mentioned=[],
    )
    return StructuredDoc(
        frontmatter=fm,
        tldr="t",
        takeaways=[],
        concepts=[],
        references=[],
        quotes=[],
        sections=[],
        open_questions=[],
        speaker_mappings=speaker_mappings,
    )


def _seg(start: float, end: float, text: str, speaker: str | None) -> Segment:
    words = [Word(text=text, start=start, end=end, speaker=speaker)]
    return Segment(start=start, end=end, text=text, speaker=speaker, words=words)


class TestTranscriptSection:
    def test_section_header(self) -> None:
        d = _doc({})
        t = Transcript(
            language="en",
            duration_s=10.0,
            backend="openai_transcribe",
            model_id="m",
            chunked=False,
            segments=[_seg(0.0, 10.0, "hello", None)],
            speakers=[],
        )
        md = render(d, t)
        assert "## Full Cleaned Transcript" in md

    def test_timestamp_marker_at_paragraph_start(self) -> None:
        d = _doc({})
        t = Transcript(
            language="en",
            duration_s=10.0,
            backend="openai_transcribe",
            model_id="m",
            chunked=False,
            segments=[_seg(0.0, 10.0, "hello", None)],
            speakers=[],
        )
        md = render(d, t)
        assert "**[00:00]**" in md


class TestSpeakerNameSubstitution:
    def test_speaker_name_substituted(self) -> None:
        d = _doc({"SPEAKER_00": "Andrew Huberman"})
        t = Transcript(
            language="en",
            duration_s=10.0,
            backend="openai_transcribe",
            model_id="m",
            chunked=False,
            segments=[_seg(0.0, 10.0, "Welcome.", "SPEAKER_00")],
            speakers=["SPEAKER_00"],
        )
        md = render(d, t)
        assert "Andrew Huberman: Welcome." in md
        assert "SPEAKER_00" not in md

    def test_unmapped_speaker_label_preserved(self) -> None:
        d = _doc({})  # no map
        t = Transcript(
            language="en",
            duration_s=10.0,
            backend="openai_transcribe",
            model_id="m",
            chunked=False,
            segments=[_seg(0.0, 10.0, "Welcome.", "SPEAKER_00")],
            speakers=["SPEAKER_00"],
        )
        md = render(d, t)
        assert "SPEAKER_00: Welcome." in md


class TestParagraphGrouping:
    def test_speaker_change_forces_new_paragraph(self) -> None:
        d = _doc({"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})
        t = Transcript(
            language="en",
            duration_s=20.0,
            backend="openai_transcribe",
            model_id="m",
            chunked=False,
            segments=[
                _seg(0.0, 5.0, "Hello.", "SPEAKER_00"),
                _seg(5.5, 10.0, "Hi back.", "SPEAKER_01"),
                _seg(10.5, 20.0, "And then.", "SPEAKER_00"),
            ],
            speakers=["SPEAKER_00", "SPEAKER_01"],
        )
        md = render(d, t)
        # Each speaker change -> its own paragraph with its own [mm:ss] marker
        assert "**[00:00]** Alice: Hello." in md
        assert "**[00:05]** Bob: Hi back." in md
        assert "**[00:10]** Alice: And then." in md

    def test_same_speaker_under_60s_grouped(self) -> None:
        d = _doc({})
        t = Transcript(
            language="en",
            duration_s=30.0,
            backend="openai_transcribe",
            model_id="m",
            chunked=False,
            segments=[
                _seg(0.0, 10.0, "First.", None),
                _seg(10.0, 20.0, "Second.", None),
                _seg(20.0, 30.0, "Third.", None),
            ],
            speakers=[],
        )
        md = render(d, t)
        # All three should be in one paragraph (same speaker None, within 60s)
        assert "**[00:00]**" in md
        assert "**[00:10]**" not in md
        assert "**[00:20]**" not in md
        # Joined text
        assert "First. Second. Third." in md

    def test_60s_boundary_forces_new_paragraph(self) -> None:
        d = _doc({})
        t = Transcript(
            language="en",
            duration_s=120.0,
            backend="openai_transcribe",
            model_id="m",
            chunked=False,
            segments=[
                _seg(0.0, 50.0, "First.", None),
                _seg(50.0, 65.0, "Second.", None),  # crosses 60s
                _seg(65.0, 120.0, "Third.", None),
            ],
            speakers=[],
        )
        md = render(d, t)
        # Block 1: [00:00] First. Second.
        # Block 2: [01:05] Third.  (next segment after block 1 boundary)
        assert "**[00:00]** First. Second." in md
        assert "**[01:05]** Third." in md
