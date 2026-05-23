"""Golden test: render(fixture_doc, fixture_transcript) == fixture_markdown byte-for-byte."""

import json
from pathlib import Path

from yt2md.models import Segment, StructuredDoc, Transcript, Word
from yt2md.stages.render import render


def test_golden(fixtures_dir: Path) -> None:
    doc = StructuredDoc.model_validate(
        json.loads((fixtures_dir / "structured" / "sample_doc.json").read_text(encoding="utf-8"))
    )
    transcript = Transcript(
        language="en",
        duration_s=10.0,
        backend="openai_transcribe",
        model_id="gpt-4o-transcribe",
        chunked=False,
        segments=[
            Segment(
                start=0.0,
                end=10.0,
                text="Welcome to the show.",
                speaker="SPEAKER_00",
                words=[Word(text="Welcome", start=0.0, end=10.0, speaker="SPEAKER_00")],
            ),
        ],
        speakers=["SPEAKER_00"],
    )
    expected = (fixtures_dir / "markdown" / "sample_doc.md").read_text(encoding="utf-8")
    actual = render(doc, transcript)
    assert actual == expected
