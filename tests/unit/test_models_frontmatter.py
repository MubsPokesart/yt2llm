"""Tests for the Frontmatter model."""

from datetime import date

import pytest
from pydantic import ValidationError

from yt2md.models import Frontmatter


class TestFrontmatter:
    def test_full(self) -> None:
        fm = Frontmatter(
            title="Dopamine, Motivation & Drive",
            channel="Huberman Lab",
            url="https://www.youtube.com/watch?v=abc",
            video_id="abc",
            published=date(2024, 3, 15),
            duration_seconds=5025,
            captured_at=date(2026, 5, 23),
            schema_version=1,
            genre="podcast",
            speakers=["Andrew Huberman"],
            topics=["dopamine", "motivation"],
            people_mentioned=["Robert Sapolsky"],
            works_mentioned=["The Molecule of More"],
        )
        assert fm.schema_version == 1
        assert fm.genre == "podcast"

    @pytest.mark.parametrize(
        "genre",
        ["podcast", "lecture", "tutorial", "talk", "interview", "other"],
    )
    def test_genre_enum(self, genre: str) -> None:
        fm = Frontmatter(
            title="t",
            channel="c",
            url="u",
            video_id="v",
            published=date(2025, 1, 1),
            duration_seconds=1,
            captured_at=date(2025, 1, 1),
            schema_version=1,
            genre=genre,  # type: ignore[arg-type]
            speakers=[],
            topics=[],
            people_mentioned=[],
            works_mentioned=[],
        )
        assert fm.genre == genre

    def test_invalid_genre_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Frontmatter(
                title="t",
                channel="c",
                url="u",
                video_id="v",
                published=date(2025, 1, 1),
                duration_seconds=1,
                captured_at=date(2025, 1, 1),
                schema_version=1,
                genre="movie",  # type: ignore[arg-type]
                speakers=[],
                topics=[],
                people_mentioned=[],
                works_mentioned=[],
            )
