"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yt2md.models import Transcript, VideoMetadata

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the tests/fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def short_solo_transcript() -> Transcript:
    """3-segment solo transcript with one filler ('Uh,'). Loaded from JSON fixture."""
    data = json.loads(
        (FIXTURES_DIR / "transcripts" / "short_solo.json").read_text(encoding="utf-8")
    )
    return Transcript.model_validate(data)


@pytest.fixture
def multi_speaker_transcript() -> Transcript:
    """2-speaker transcript with 96/4% duration split (above 95% collapse threshold)."""
    data = json.loads(
        (FIXTURES_DIR / "transcripts" / "multi_speaker.json").read_text(encoding="utf-8")
    )
    return Transcript.model_validate(data)


@pytest.fixture
def huberman_metadata() -> VideoMetadata:
    """Sample VideoMetadata mimicking a Huberman Lab episode."""
    data = json.loads(
        (FIXTURES_DIR / "metadata" / "huberman_sample.json").read_text(encoding="utf-8")
    )
    return VideoMetadata.model_validate(data)
