"""Shared fixtures for integration tests."""

from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from yt2md.config import Config
from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    Frontmatter,
    Segment,
    StructuredDoc,
    Takeaway,
    Transcript,
    VideoMetadata,
    Word,
)


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        google_api_key="g",  # type: ignore[arg-type]
        openai_api_key="o",  # type: ignore[arg-type]
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )


def _make_metadata() -> VideoMetadata:
    return VideoMetadata(
        video_id="abc123",
        url="https://www.youtube.com/watch?v=abc123",
        title="Test Episode",
        channel="Test Channel",
        channel_id="UC",
        published_date=date(2024, 3, 15),
        duration_s=10.0,
        description="",
        chapters=[],
        tags=[],
        language="en",
    )


def _make_transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_s=10.0,
        backend="openai_transcribe",
        model_id="gpt-4o-transcribe",
        chunked=False,
        segments=[
            Segment(
                start=0.0,
                end=10.0,
                text="hello world.",
                speaker="SPEAKER_00",
                words=[
                    Word(text="hello", start=0.0, end=5.0, speaker="SPEAKER_00"),
                    Word(text="world.", start=5.0, end=10.0, speaker="SPEAKER_00"),
                ],
            ),
        ],
        speakers=["SPEAKER_00"],
    )


def _make_structured_doc() -> StructuredDoc:
    return StructuredDoc(
        frontmatter=Frontmatter(
            title="Test Episode",
            channel="Test Channel",
            url="https://www.youtube.com/watch?v=abc123",
            video_id="abc123",
            published=date(2024, 3, 15),
            duration_seconds=10,
            captured_at=date(2026, 5, 23),
            schema_version=CURRENT_SCHEMA_VERSION,
            genre="podcast",
            speakers=["Alice"],
            topics=[],
            people_mentioned=[],
            works_mentioned=[],
        ),
        tldr="TLDR sentence.",
        takeaways=[
            Takeaway(text="One.", timestamp_s=0.0),
            Takeaway(text="Two.", timestamp_s=2.0),
            Takeaway(text="Three.", timestamp_s=4.0),
        ],
        concepts=[],
        references=[],
        quotes=[],
        sections=[],
        open_questions=[],
        speaker_name_map={"SPEAKER_00": "Alice"},
    )


@pytest.fixture
def patched_stages(cfg: Config) -> Any:
    """Patch every external-touching stage."""
    metadata = _make_metadata()
    transcript = _make_transcript()
    doc = _make_structured_doc()

    source_audio = cfg.cache_dir / "abc123" / "source_audio.m4a"
    source_audio.parent.mkdir(parents=True, exist_ok=True)
    source_audio.write_bytes(b"\x00" * 100)

    def fake_compress(*, source: Path, destination: Path, cfg: Config) -> None:  # noqa: ARG001
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x00" * 50)

    with (
        patch("yt2md.pipeline.download") as dl,
        patch("yt2md.pipeline.compress") as cmp,
        patch("yt2md.pipeline.transcribe") as tx,
        patch("yt2md.pipeline.structure") as st,
    ):
        dl.return_value = (source_audio, metadata, {"id": "abc123"})
        cmp.side_effect = fake_compress
        tx.return_value = (transcript, [{"language": "en"}])
        st.return_value = doc

        yield {"dl": dl, "cmp": cmp, "tx": tx, "st": st}
