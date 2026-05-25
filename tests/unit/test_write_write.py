"""Tests for the write() function: atomic write + collision handling."""

from datetime import date
from pathlib import Path

from yt2md.models import CURRENT_SCHEMA_VERSION, Frontmatter, StructuredDoc, Takeaway
from yt2md.stages.write import write


def _doc(title: str = "T", channel: str = "C", video_id: str = "vid") -> StructuredDoc:
    fm = Frontmatter(
        title=title,
        channel=channel,
        url=f"https://www.youtube.com/watch?v={video_id}",
        video_id=video_id,
        published=date(2024, 3, 15),
        duration_seconds=10,
        captured_at=date(2026, 5, 23),
        schema_version=CURRENT_SCHEMA_VERSION,
        genre="podcast",
        speakers=[],
        topics=[],
        people_mentioned=[],
        works_mentioned=[],
    )
    return StructuredDoc(
        frontmatter=fm,
        tldr="t",
        takeaways=[Takeaway(text="x", timestamp_s=0.0)],
        concepts=[],
        references=[],
        quotes=[],
        sections=[],
        open_questions=[],
        speaker_mappings=[],
    )


class TestWriteHappyPath:
    def test_writes_to_correct_filename(self, tmp_path: Path) -> None:
        d = _doc(title="Dopamine", channel="Huberman Lab")
        path = write(markdown="hello", doc=d, output_dir=tmp_path)
        assert path == tmp_path / "2024-03-15__huberman-lab__dopamine.md"
        assert path.read_text(encoding="utf-8") == "hello"

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        d = _doc()
        out_dir = tmp_path / "nested" / "out"
        path = write(markdown="x", doc=d, output_dir=out_dir)
        assert path.parent == out_dir
        assert path.exists()

    def test_no_tmp_left_behind(self, tmp_path: Path) -> None:
        d = _doc()
        write(markdown="x", doc=d, output_dir=tmp_path)
        leftover = list(tmp_path.glob("*.tmp"))
        assert leftover == []


class TestCollision:
    def test_collision_with_different_video_id_appends_suffix(self, tmp_path: Path) -> None:
        # Pre-create a file with a different video_id's frontmatter
        existing = tmp_path / "2024-03-15__huberman-lab__dopamine.md"
        existing.write_text("---\nvideo_id: OTHER\n---\n", encoding="utf-8")

        d = _doc(title="Dopamine", channel="Huberman Lab", video_id="MINE")
        path = write(markdown="hello", doc=d, output_dir=tmp_path)
        assert path == tmp_path / "2024-03-15__huberman-lab__dopamine__MINE.md"
        assert path.read_text(encoding="utf-8") == "hello"
        # Existing file untouched
        assert existing.read_text(encoding="utf-8") == "---\nvideo_id: OTHER\n---\n"

    def test_collision_with_same_video_id_overwrites(self, tmp_path: Path) -> None:
        # Same video_id -> overwrite (idempotency check would normally short-circuit
        # before reaching write(); this test exercises the lower-level behavior).
        existing = tmp_path / "2024-03-15__huberman-lab__dopamine.md"
        existing.write_text("---\nvideo_id: vid\n---\nOLD", encoding="utf-8")

        d = _doc(title="Dopamine", channel="Huberman Lab", video_id="vid")
        path = write(markdown="NEW", doc=d, output_dir=tmp_path)
        assert path == existing
        assert path.read_text(encoding="utf-8") == "NEW"
