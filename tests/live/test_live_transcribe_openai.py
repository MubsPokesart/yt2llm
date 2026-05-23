"""Live test for OpenAI transcribe backend. Skipped without -m live + OPENAI_API_KEY."""

import os
from datetime import date
from pathlib import Path

import pytest

from yt2md.config import Config
from yt2md.models import VideoMetadata
from yt2md.stages.transcribe_backends.openai import transcribe_openai


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_live_transcribe_30s(fixtures_dir: Path, tmp_path: Path) -> None:
    audio = fixtures_dir / "audio" / "short_speech_30s.opus"
    if not audio.exists():
        pytest.skip(f"Fixture audio missing: {audio}")

    cfg = Config(
        google_api_key="g",  # type: ignore[arg-type]
        openai_api_key=os.environ["OPENAI_API_KEY"],  # type: ignore[arg-type]
        cache_dir=tmp_path,
        output_dir=tmp_path / "out",
    )
    meta = VideoMetadata(
        video_id="x",
        url="https://www.youtube.com/watch?v=x",
        title="Test",
        channel="Test",
        channel_id="UC",
        published_date=date(2025, 1, 1),
        duration_s=30.0,
        description="",
        chapters=[],
        tags=[],
        language="en",
    )
    transcript, _raw = transcribe_openai(audio, meta, cfg=cfg)
    assert transcript.duration_s > 0
    assert len(transcript.segments) >= 1
    assert any(w.text for s in transcript.segments for w in s.words)
