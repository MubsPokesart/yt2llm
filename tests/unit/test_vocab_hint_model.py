"""Tests for the VocabularyHints dataclass and VOCAB_HINT_VERSION constant."""

from yt2md.vocab_hint import VOCAB_HINT_VERSION, VocabularyHints


class TestVocabHintVersion:
    def test_is_positive_int(self) -> None:
        assert isinstance(VOCAB_HINT_VERSION, int)
        assert VOCAB_HINT_VERSION >= 1


class TestVocabularyHints:
    def test_empty(self) -> None:
        h = VocabularyHints(
            people=[],
            works=[],
            concepts=[],
            organizations=[],
            channel="",
            title="",
        )
        assert h.people == []

    def test_populated(self) -> None:
        h = VocabularyHints(
            people=["Andrew Huberman"],
            works=["The Molecule of More"],
            concepts=["dopamine"],
            organizations=["Stanford"],
            channel="Huberman Lab",
            title="Dopamine",
        )
        assert h.people == ["Andrew Huberman"]
        assert h.channel == "Huberman Lab"
