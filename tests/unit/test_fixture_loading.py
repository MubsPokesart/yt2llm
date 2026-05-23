"""Smoke tests: fixtures load into Pydantic models without error."""

import pytest

from yt2md.models import Transcript, VideoMetadata


def test_short_solo_transcript_loads(short_solo_transcript: Transcript) -> None:
    assert short_solo_transcript.duration_s == pytest.approx(30.0)
    assert short_solo_transcript.speakers == ["SPEAKER_00"]
    assert len(short_solo_transcript.segments) == 3
    # Segment 2 starts with the filler "Uh,"
    assert short_solo_transcript.segments[1].words[0].text == "Uh,"


def test_multi_speaker_transcript_loads(multi_speaker_transcript: Transcript) -> None:
    assert multi_speaker_transcript.duration_s == pytest.approx(100.0)
    assert multi_speaker_transcript.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert len(multi_speaker_transcript.segments) == 2


def test_huberman_metadata_loads(huberman_metadata: VideoMetadata) -> None:
    assert huberman_metadata.title == "Dopamine, Motivation & Drive"
    assert huberman_metadata.channel == "Huberman Lab"
    assert len(huberman_metadata.chapters) == 3
