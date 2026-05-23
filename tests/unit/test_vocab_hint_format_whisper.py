"""Tests for vocab_hint.format_for_whisper — natural-sentence style."""

from yt2md.vocab_hint import VocabularyHints, format_for_whisper


def _hints(**kwargs: object) -> VocabularyHints:
    defaults: dict[str, object] = {
        "people": [],
        "works": [],
        "concepts": [],
        "organizations": [],
        "channel": "C",
        "title": "T",
    }
    defaults.update(kwargs)
    return VocabularyHints(**defaults)  # type: ignore[arg-type]


class TestFormatForWhisper:
    def test_starts_with_transcript_framing(self) -> None:
        h = _hints(people=["Andrew Huberman"], channel="Huberman Lab")
        out = format_for_whisper(h)
        # Whisper mimics style. Frame as a transcript description.
        assert "transcript" in out.lower()

    def test_includes_channel_in_sentence(self) -> None:
        h = _hints(channel="Huberman Lab")
        out = format_for_whisper(h)
        assert "Huberman Lab" in out

    def test_includes_people_in_sentence(self) -> None:
        h = _hints(people=["Andrew Huberman"], channel="Huberman Lab")
        out = format_for_whisper(h)
        assert "Andrew Huberman" in out

    def test_includes_works(self) -> None:
        h = _hints(works=["The Molecule of More"])
        out = format_for_whisper(h)
        assert "The Molecule of More" in out

    def test_preserves_capitalization(self) -> None:
        h = _hints(concepts=["GPT-4", "PyTorch"])
        out = format_for_whisper(h)
        # Whisper picks up capitalization from prompt; must preserve exactly.
        assert "GPT-4" in out
        assert "PyTorch" in out
        assert "gpt-4" not in out
        assert "pytorch" not in out

    def test_ends_with_period(self) -> None:
        h = _hints(people=["A"], channel="C", title="T")
        out = format_for_whisper(h).strip()
        assert out.endswith(".")

    def test_empty_hints_returns_string(self) -> None:
        h = _hints()
        out = format_for_whisper(h)
        assert isinstance(out, str)
        assert out.strip()  # non-empty
