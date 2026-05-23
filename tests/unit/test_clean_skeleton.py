"""Tests for the clean stage skeleton and module constants."""

from yt2md.models import Transcript
from yt2md.stages.clean import CLEANER_VERSION, HARD_FILLERS, clean


class TestConstants:
    def test_cleaner_version_is_positive_int(self) -> None:
        assert isinstance(CLEANER_VERSION, int)
        assert CLEANER_VERSION >= 1

    def test_hard_fillers_canonical_set(self) -> None:
        # PodcastFillers-derived; covers ~96% of empirically annotated fillers.
        # Excludes "mm"/"mhm" (agreement sounds) and "you know"/"like"/"I mean"
        # (context-dependent, removal risks meaning loss).
        assert frozenset({"uh", "um", "uhm", "er", "ah"}) == HARD_FILLERS


class TestCleanIdentity:
    def test_clean_returns_transcript(self, short_solo_transcript: Transcript) -> None:
        result = clean(short_solo_transcript)
        assert isinstance(result, Transcript)
