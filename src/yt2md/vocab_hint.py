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

from dataclasses import dataclass, field

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
