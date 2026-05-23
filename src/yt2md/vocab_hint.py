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

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .models import VideoMetadata

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


def extract_hints(meta: VideoMetadata) -> VocabularyHints:
    """Extract a categorized VocabularyHints from video metadata.

    Sources (priority order): title > channel > chapter titles > first 500 chars
    of description. URLs are stripped before scanning.
    """
    desc_excerpt = _strip_urls(meta.description)[:500]
    sources = [meta.title, *(c.title for c in meta.chapters), desc_excerpt]

    people = _dedup_ordered(match for src in sources for match in _TITLE_CASE_PATTERN.findall(src))

    return VocabularyHints(
        people=people,
        works=[],
        concepts=[],
        organizations=[],
        channel=meta.channel,
        title=meta.title,
    )


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
