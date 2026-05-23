"""Property tests: token budget is always respected regardless of input."""

import tiktoken
from hypothesis import given
from hypothesis import strategies as st

from yt2md.vocab_hint import VocabularyHints, format_for_openai, format_for_whisper

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


# Reasonable strategy: short strings, bounded lists.
_short_str = st.text(min_size=0, max_size=40)
_str_list = st.lists(_short_str, min_size=0, max_size=20)


@given(
    people=_str_list,
    works=_str_list,
    concepts=_str_list,
    organizations=_str_list,
    channel=_short_str,
    title=_short_str,
)
def test_openai_format_respects_budget(
    people: list[str],
    works: list[str],
    concepts: list[str],
    organizations: list[str],
    channel: str,
    title: str,
) -> None:
    h = VocabularyHints(
        people=people,
        works=works,
        concepts=concepts,
        organizations=organizations,
        channel=channel,
        title=title,
    )
    out = format_for_openai(h, max_tokens=50)
    assert _count_tokens(out) <= 50


@given(
    people=_str_list,
    works=_str_list,
    concepts=_str_list,
    organizations=_str_list,
    channel=_short_str,
    title=_short_str,
)
def test_whisper_format_respects_budget(
    people: list[str],
    works: list[str],
    concepts: list[str],
    organizations: list[str],
    channel: str,
    title: str,
) -> None:
    h = VocabularyHints(
        people=people,
        works=works,
        concepts=concepts,
        organizations=organizations,
        channel=channel,
        title=title,
    )
    out = format_for_whisper(h, max_tokens=50)
    assert _count_tokens(out) <= 50
