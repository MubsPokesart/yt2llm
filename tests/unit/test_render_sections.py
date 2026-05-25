"""Tests for Concepts, References (with emoji), Notable Quotes sections."""

from datetime import date

import pytest

from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    Concept,
    Frontmatter,
    Quote,
    Reference,
    StructuredDoc,
    Transcript,
)
from yt2md.stages.render import render


def _doc(
    concepts: list[Concept] | None = None,
    references: list[Reference] | None = None,
    quotes: list[Quote] | None = None,
) -> StructuredDoc:
    fm = Frontmatter(
        title="T",
        channel="C",
        url="https://www.youtube.com/watch?v=v",
        video_id="v",
        published=date(2025, 1, 1),
        duration_seconds=60,
        captured_at=date(2026, 5, 23),
        schema_version=CURRENT_SCHEMA_VERSION,
        genre="podcast",
        speakers=["A"],
        topics=[],
        people_mentioned=[],
        works_mentioned=[],
    )
    return StructuredDoc(
        frontmatter=fm,
        tldr="t",
        takeaways=[],
        concepts=concepts or [],
        references=references or [],
        quotes=quotes or [],
        sections=[],
        open_questions=[],
        speaker_mappings=[],
    )


@pytest.fixture
def empty_transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_s=60.0,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=[],
        speakers=["A"],
    )


class TestConcepts:
    def test_concept_rendered(self, empty_transcript: Transcript) -> None:
        d = _doc(
            concepts=[
                Concept(
                    name="Reward Prediction Error",
                    definition="Gap between expected and actual.",
                    timestamp_s=510.0,
                ),
            ]
        )
        md = render(d, empty_transcript)
        assert "## Concepts & Definitions" in md
        assert "Reward Prediction Error" in md
        assert "Gap between expected and actual." in md
        assert "[08:30]" in md


class TestReferences:
    @pytest.mark.parametrize(
        ("kind", "expected_emoji"),
        [
            ("book", "📚"),
            ("paper", "📄"),
            ("person", "👤"),
            ("tool", "🛠"),
            ("video", "🎬"),
            ("other", "🔗"),
        ],
    )
    def test_emoji_prefix(
        self, empty_transcript: Transcript, kind: str, expected_emoji: str
    ) -> None:
        d = _doc(
            references=[
                Reference(kind=kind, name="X", context="c", timestamp_s=0.0),  # type: ignore[arg-type]
            ]
        )
        md = render(d, empty_transcript)
        assert expected_emoji in md

    def test_reference_text(self, empty_transcript: Transcript) -> None:
        d = _doc(
            references=[
                Reference(
                    kind="book",
                    name="The Molecule of More",
                    context="Cited as accessible primer",
                    timestamp_s=902.0,
                ),
            ]
        )
        md = render(d, empty_transcript)
        assert "## References Mentioned" in md
        assert "The Molecule of More" in md
        assert "Cited as accessible primer" in md


class TestQuotes:
    def test_quote_rendered(self, empty_transcript: Transcript) -> None:
        d = _doc(
            quotes=[
                Quote(text="Pursuit, not pleasure.", speaker="Andrew Huberman", timestamp_s=754.0),
            ]
        )
        md = render(d, empty_transcript)
        assert "## Notable Quotes" in md
        # Block-quote prefix
        assert "> Pursuit, not pleasure." in md
        assert "— Andrew Huberman" in md
        assert "[12:34]" in md
