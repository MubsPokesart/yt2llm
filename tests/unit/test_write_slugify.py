"""Tests for slugify(): ASCII, hyphenated, max 80 chars."""

from yt2md.stages.write import slugify


class TestSlugify:
    def test_basic_lowercase(self) -> None:
        assert slugify("Hello World") == "hello-world"

    def test_strips_punctuation(self) -> None:
        assert slugify("Dopamine, Motivation & Drive!") == "dopamine-motivation-drive"

    def test_collapses_whitespace(self) -> None:
        assert slugify("hello   world   here") == "hello-world-here"

    def test_max_80_chars(self) -> None:
        s = "a" * 100
        out = slugify(s)
        assert len(out) <= 80

    def test_ascii_only(self) -> None:
        # Unicode chars stripped (or transliterated to ASCII-friendly)
        out = slugify("café résumé")
        assert out.isascii()

    def test_no_leading_or_trailing_hyphens(self) -> None:
        out = slugify("---hello---")
        assert not out.startswith("-")
        assert not out.endswith("-")

    def test_no_consecutive_hyphens(self) -> None:
        out = slugify("hello  --  world")
        assert "--" not in out

    def test_empty_input_returns_empty(self) -> None:
        assert not slugify("")

    def test_only_punctuation_returns_empty(self) -> None:
        assert not slugify("!@#$%^&*()")
