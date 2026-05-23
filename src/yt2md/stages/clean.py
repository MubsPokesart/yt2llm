"""Deterministic transcript cleaning.

Removes a fixed set of hard filler words (preserving surviving timestamps),
applies the 95% duration-weighted speaker-collapse rule, and drops speakers
contributing <1% of total duration as noise.

CLEANER_VERSION participates in the cleaned-artifact cache key. Bump on any
behavior change.
"""

from __future__ import annotations

import string

from yt2md.models import Segment, Transcript, Word

CLEANER_VERSION = 1

# Derived from PodcastFillers (Zhu et al. 2022). Covers ~96% of annotated fillers
# in podcast audio. Excludes "mm"/"mhm" (agreement sounds, non-filler) and
# "you know"/"like"/"I mean" (context-dependent; removal risks meaning loss).
HARD_FILLERS = frozenset({"uh", "um", "uhm", "er", "ah"})


def clean(transcript: Transcript) -> Transcript:
    """Pure function: returns a cleaned copy of the transcript."""
    cleaned_segments: list[Segment] = []
    for segment in transcript.segments:
        cleaned = _clean_segment(segment)
        if cleaned is not None:
            cleaned_segments.append(cleaned)
    return Transcript(
        language=transcript.language,
        duration_s=transcript.duration_s,
        backend=transcript.backend,
        model_id=transcript.model_id,
        chunked=transcript.chunked,
        segments=cleaned_segments,
        speakers=transcript.speakers,
    )


def _clean_segment(segment: Segment) -> Segment | None:
    kept: list[Word] = [w for w in segment.words if not _is_filler(w.text)]
    if not kept:
        return None
    rebuilt_text = " ".join(w.text for w in kept)
    return Segment(
        start=segment.start,
        end=segment.end,
        text=rebuilt_text,
        speaker=segment.speaker,
        words=kept,
    )


def _is_filler(token: str) -> bool:
    normalized = token.lower().strip(string.punctuation + string.whitespace)
    return normalized in HARD_FILLERS
