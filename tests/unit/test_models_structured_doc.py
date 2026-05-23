"""Tests for the StructuredDoc root model + CURRENT_SCHEMA_VERSION constant."""

from datetime import date

from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    Concept,
    DetailedSection,
    Frontmatter,
    Quote,
    Reference,
    StructuredDoc,
    Takeaway,
)


def _make_frontmatter() -> Frontmatter:
    return Frontmatter(
        title="t",
        channel="c",
        url="u",
        video_id="v",
        published=date(2025, 1, 1),
        duration_seconds=10,
        captured_at=date(2025, 1, 1),
        schema_version=CURRENT_SCHEMA_VERSION,
        genre="podcast",
        speakers=["A"],
        topics=[],
        people_mentioned=[],
        works_mentioned=[],
    )


class TestStructuredDoc:
    def test_minimal(self) -> None:
        doc = StructuredDoc(
            frontmatter=_make_frontmatter(),
            tldr="Short.",
            takeaways=[Takeaway(text="x", timestamp_s=0.0)],
            concepts=[],
            references=[],
            quotes=[],
            sections=[],
            open_questions=[],
            speaker_name_map={"SPEAKER_00": "A"},
        )
        assert doc.tldr == "Short."

    def test_round_trip_json(self) -> None:
        doc = StructuredDoc(
            frontmatter=_make_frontmatter(),
            tldr="Hello.",
            takeaways=[Takeaway(text="x", timestamp_s=0.0)],
            concepts=[Concept(name="N", definition="D", timestamp_s=1.0)],
            references=[Reference(kind="book", name="B", context="c", timestamp_s=2.0)],
            quotes=[Quote(text="q", speaker="A", timestamp_s=3.0)],
            sections=[DetailedSection(heading="H", body="B", timestamp_s=4.0)],
            open_questions=["?"],
            speaker_name_map={"SPEAKER_00": "A"},
        )
        back = StructuredDoc.model_validate_json(doc.model_dump_json())
        assert back == doc


class TestSchemaVersion:
    def test_current_version_is_positive_int(self) -> None:
        assert isinstance(CURRENT_SCHEMA_VERSION, int)
        assert CURRENT_SCHEMA_VERSION >= 1
