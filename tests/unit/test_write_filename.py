"""Tests for build_filename: {published}__{channel-slug}__{title-slug}.md format."""

from datetime import date

from yt2md.models import CURRENT_SCHEMA_VERSION, Frontmatter
from yt2md.stages.write import build_filename


def _fm(
    title: str = "Title",
    channel: str = "Channel",
    published: date | None = None,
) -> Frontmatter:
    return Frontmatter(
        title=title,
        channel=channel,
        url="u",
        video_id="vid",
        published=published or date(2024, 3, 15),
        duration_seconds=10,
        captured_at=date(2026, 5, 23),
        schema_version=CURRENT_SCHEMA_VERSION,
        genre="podcast",
        speakers=[],
        topics=[],
        people_mentioned=[],
        works_mentioned=[],
    )


class TestBuildFilename:
    def test_format(self) -> None:
        fm = _fm(title="Dopamine, Motivation & Drive", channel="Huberman Lab")
        assert build_filename(fm) == "2024-03-15__huberman-lab__dopamine-motivation-drive.md"

    def test_uses_published_not_captured(self) -> None:
        fm = _fm(published=date(2020, 1, 1))
        out = build_filename(fm)
        assert out.startswith("2020-01-01__")

    def test_slugifies_channel(self) -> None:
        fm = _fm(channel="The Tim Ferriss Show!")
        out = build_filename(fm)
        assert "__the-tim-ferriss-show__" in out

    def test_ends_with_md(self) -> None:
        fm = _fm()
        assert build_filename(fm).endswith(".md")
