"""Live test for Gemini structure stage. Skipped without -m live + GOOGLE_API_KEY."""

import json
import os
from pathlib import Path

import pytest

from yt2md.config import Config
from yt2md.models import Transcript, VideoMetadata
from yt2md.stages.structure import structure


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
def test_live_structure_small_transcript(fixtures_dir: Path, tmp_path: Path) -> None:
    transcript = Transcript.model_validate(
        json.loads((fixtures_dir / "transcripts" / "short_solo.json").read_text(encoding="utf-8"))
    )
    metadata = VideoMetadata.model_validate(
        json.loads((fixtures_dir / "metadata" / "huberman_sample.json").read_text(encoding="utf-8"))
    )
    # Resync to transcript's actual duration
    metadata = metadata.model_copy(update={"duration_s": transcript.duration_s})

    cfg = Config(
        google_api_key=os.environ["GOOGLE_API_KEY"],  # type: ignore[arg-type]
        cache_dir=tmp_path,
    )
    doc = structure(transcript, metadata, cfg=cfg)
    assert len(doc.takeaways) >= 3
    assert doc.tldr.strip()
    assert doc.frontmatter.title == metadata.title
