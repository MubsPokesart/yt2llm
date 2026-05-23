"""Vocabulary hint construction for transcription backends.

The OpenAI gpt-4o-transcribe `prompt` param and faster-whisper `initial_prompt` param
accept ~224 tokens of vocabulary/style biasing. Empirically validated by PodcastFillers
and OpenAI cookbook guidance:
  - OpenAI: comma-separated glossary, ~"Important words: A, B, C."
  - Whisper: natural sentence form ("This is a transcript of ... discussing ...")
    so the model mimics style and capitalization.

VOCAB_HINT_VERSION is bumped manually when extraction logic or per-backend formatting
changes meaning. Bumping invalidates the transcript cache key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from collections.abc import Iterable

    from yt2md.models import VideoMetadata

VOCAB_HINT_VERSION = 1


@dataclass(frozen=True)
class VocabularyHints:
    """Structured vocabulary extracted from video metadata.

    Categorization powers per-backend formatting:
      - people: subject/object in Whisper sentences; first in OpenAI glossary
      - works: italicized in Whisper sentences; second in OpenAI glossary
      - organizations: locations/affiliations
      - concepts: topics, technical terms, acronyms
      - channel + title: always included as opening context
    """

    people: list[str] = field(default_factory=list)
    works: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    channel: str = ""
    title: str = ""


# Title Case sequence: capitalized word followed by 1-3 more capitalized words.
# Tightened to require >=2 words so single-word capitalized common nouns
# ("Welcome") don't pollute the people list. Wrapped in a lookahead so findall
# returns overlapping matches: "Featuring Andrew Huberman" yields both the full
# match and the inner "Andrew Huberman".
_TITLE_CASE_PATTERN = re.compile(r"(?=\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b)")

# Quoted phrases (single, double, or smart quotes) — likely titles of works.
_QUOTED_PATTERN = re.compile(r'["“]([^"”]{2,80})["”]')

# All-caps acronyms, 2-5 chars. Single letters and longer strings excluded
# (longer all-caps is usually shouting or noise; single letters are stop-words).
_ACRONYM_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")

# CamelCase: starts with capital, has at least one lowercase, then at least one capital
# OR alphanumeric with internal digit/hyphen ("GPT-4", "Claude-3").
_CAMELCASE_PATTERN = re.compile(r"\b([A-Z][a-z]+[A-Z][A-Za-z0-9]*|[A-Z][A-Za-z]+-?\d+)\b")


def extract_hints(meta: VideoMetadata) -> VocabularyHints:
    """Extract a categorized VocabularyHints from video metadata.

    Sources (priority order): title > channel > chapter titles > first 500 chars
    of description. URLs are stripped before scanning.
    """
    desc_excerpt = _strip_urls(meta.description)[:500]
    sources = [meta.title, *(c.title for c in meta.chapters), desc_excerpt]
    combined = "\n".join(sources)

    works = _dedup_ordered(_QUOTED_PATTERN.findall(combined))
    acronyms = _dedup_ordered(_ACRONYM_PATTERN.findall(combined))
    camel = _dedup_ordered(_CAMELCASE_PATTERN.findall(combined))
    concepts = _dedup_ordered([*acronyms, *camel])
    people = _dedup_ordered(_TITLE_CASE_PATTERN.findall(combined))

    return VocabularyHints(
        people=people,
        works=works,
        concepts=concepts,
        organizations=[],
        channel=meta.channel,
        title=meta.title,
    )


DEFAULT_TOKEN_BUDGET = 220


def format_for_openai(hints: VocabularyHints, *, max_tokens: int = DEFAULT_TOKEN_BUDGET) -> str:
    """Format hints as a comma-separated glossary for gpt-4o-transcribe's `prompt` param.

    Per OpenAI guidance: short keyword lists work better than instructions for the
    transcription `prompt` parameter. Truncates to `max_tokens` via tiktoken.
    """
    parts: list[str] = []
    if hints.title:
        parts.append(hints.title)
    if hints.channel:
        parts.append(hints.channel)
    parts.extend(hints.people)
    parts.extend(hints.works)
    parts.extend(hints.organizations)
    parts.extend(hints.concepts)
    body = ", ".join(p for p in parts if p)
    full = f"Glossary for transcription: {body}"
    return _truncate_to_tokens(full, max_tokens)


def format_for_whisper(hints: VocabularyHints, *, max_tokens: int = DEFAULT_TOKEN_BUDGET) -> str:
    """Format hints as a natural-language paragraph for Whisper's `initial_prompt`.

    Whisper mimics style/capitalization rather than following instructions.
    Sentences are constructed so the names appear in natural grammatical contexts.
    """
    sentences: list[str] = []

    opener = _build_opener(hints)
    if opener:
        sentences.append(opener)

    if hints.people:
        sentences.append(f"The speakers include {_join_oxford(hints.people)}.")

    if hints.works:
        sentences.append(f"Works referenced include {_join_oxford(hints.works)}.")

    if hints.organizations:
        sentences.append(f"Affiliations mentioned: {_join_oxford(hints.organizations)}.")

    if hints.concepts:
        sentences.append(f"Concepts discussed include {_join_oxford(hints.concepts)}.")

    if not sentences:
        # Fallback minimal sentence — Whisper expects SOME content.
        sentences.append(f"This is a transcript from {hints.channel or 'a video'}.")

    full = " ".join(sentences)
    return _truncate_to_tokens(full, max_tokens)


def _build_opener(hints: VocabularyHints) -> str:
    if hints.channel and hints.title:
        return f'This is a transcript of an episode of {hints.channel} titled "{hints.title}".'
    if hints.channel:
        return f"This is a transcript from {hints.channel}."
    if hints.title:
        return f'This is a transcript of "{hints.title}".'
    return ""


_PAIR_LENGTH = 2


def _join_oxford(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == _PAIR_LENGTH:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate `text` so the token count does not exceed `max_tokens`.

    Uses tiktoken's cl100k_base encoding (OpenAI's standard).
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    decoded: str = enc.decode(tokens[:max_tokens])
    return decoded


def _strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+", "", text)


def _dedup_ordered(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
