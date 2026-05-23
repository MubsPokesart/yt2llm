"""Tests for vocab_hint.format_for_openai — comma-separated glossary."""

from yt2md.vocab_hint import VocabularyHints, format_for_openai


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


class TestFormatForOpenAI:
    def test_starts_with_glossary_framing(self) -> None:
        h = _hints(people=["Andrew Huberman"])
        out = format_for_openai(h)
        assert out.lower().startswith("glossary")

    def test_includes_people(self) -> None:
        h = _hints(people=["Andrew Huberman", "Robert Sapolsky"])
        out = format_for_openai(h)
        assert "Andrew Huberman" in out
        assert "Robert Sapolsky" in out

    def test_includes_works(self) -> None:
        h = _hints(works=["The Molecule of More"])
        out = format_for_openai(h)
        assert "The Molecule of More" in out

    def test_includes_concepts(self) -> None:
        h = _hints(concepts=["dopamine", "GPT-4"])
        out = format_for_openai(h)
        assert "dopamine" in out
        assert "GPT-4" in out

    def test_includes_title(self) -> None:
        h = _hints(title="Dopamine and Drive")
        out = format_for_openai(h)
        assert "Dopamine and Drive" in out

    def test_includes_channel(self) -> None:
        h = _hints(channel="Huberman Lab")
        out = format_for_openai(h)
        assert "Huberman Lab" in out

    def test_comma_separated(self) -> None:
        h = _hints(people=["A", "B"], concepts=["C"])
        out = format_for_openai(h)
        # No newlines; commas everywhere
        assert "\n" not in out
        assert "," in out

    def test_empty_hints_still_returns_string(self) -> None:
        h = _hints()
        out = format_for_openai(h)
        assert isinstance(out, str)
        assert "T" in out  # title
        assert "C" in out  # channel
