"""Tests for vocab_hint.extract_hints — categorized extraction from VideoMetadata."""

from datetime import date

from yt2md.models import VideoMetadata
from yt2md.vocab_hint import extract_hints


def _meta(*, title: str = "T", channel: str = "C", description: str = "") -> VideoMetadata:
    return VideoMetadata(
        video_id="v",
        url="https://www.youtube.com/watch?v=v",
        title=title,
        channel=channel,
        channel_id="UC1",
        published_date=date(2025, 1, 1),
        duration_s=60.0,
        description=description,
        chapters=[],
        tags=[],
        language="en",
    )


class TestExtractPeople:
    def test_title_case_two_words(self) -> None:
        m = _meta(description="Featuring Andrew Huberman from Stanford.")
        h = extract_hints(m)
        assert "Andrew Huberman" in h.people

    def test_title_case_three_words(self) -> None:
        m = _meta(description="An interview with Mary Lou Jepsen.")
        h = extract_hints(m)
        assert "Mary Lou Jepsen" in h.people

    def test_lowercase_not_extracted(self) -> None:
        m = _meta(description="just regular sentence content here")
        h = extract_hints(m)
        assert h.people == []


class TestExtractTitleAndChannel:
    def test_title_present(self) -> None:
        m = _meta(title="Dopamine and Drive")
        h = extract_hints(m)
        assert h.title == "Dopamine and Drive"

    def test_channel_present(self) -> None:
        m = _meta(channel="Huberman Lab")
        h = extract_hints(m)
        assert h.channel == "Huberman Lab"


class TestExtractWorks:
    def test_double_quoted_extracted_as_work(self) -> None:
        m = _meta(description='Discussing "The Molecule of More" by Daniel Lieberman.')
        h = extract_hints(m)
        assert "The Molecule of More" in h.works

    def test_smart_quotes_extracted(self) -> None:
        m = _meta(description="Discussing “The Molecule of More” at length.")
        h = extract_hints(m)
        assert "The Molecule of More" in h.works


class TestExtractAcronyms:
    def test_short_acronyms_extracted(self) -> None:
        m = _meta(description="The fMRI scans showed ADHD signatures and DNA damage.")
        h = extract_hints(m)
        # 2-5 char all-caps tokens
        assert "ADHD" in h.concepts
        assert "DNA" in h.concepts

    def test_single_letter_not_extracted(self) -> None:
        m = _meta(description="A study of A and B groups.")
        h = extract_hints(m)
        assert "A" not in h.concepts
        assert "B" not in h.concepts

    def test_six_letter_not_extracted(self) -> None:
        m = _meta(description="SHOULDNT extract this.")
        h = extract_hints(m)
        assert "SHOULDNT" not in h.concepts


class TestExtractCamelCase:
    def test_camel_case_extracted_as_concept(self) -> None:
        m = _meta(description="Using PyTorch and TensorFlow for the experiment.")
        h = extract_hints(m)
        assert "PyTorch" in h.concepts
        assert "TensorFlow" in h.concepts

    def test_alphanumeric_with_internal_digit(self) -> None:
        m = _meta(description="GPT-4 and Claude-3 are competing models.")
        h = extract_hints(m)
        assert "GPT-4" in h.concepts


class TestDedup:
    def test_repeated_term_deduped(self) -> None:
        m = _meta(description="Andrew Huberman explained. Andrew Huberman emphasized.")
        h = extract_hints(m)
        assert h.people.count("Andrew Huberman") == 1
