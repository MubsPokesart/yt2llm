"""Deterministic transcript cleaning.

Removes a fixed set of hard filler words (preserving surviving timestamps),
applies the 95% duration-weighted speaker-collapse rule, and drops speakers
contributing <1% of total duration as noise.

CLEANER_VERSION participates in the cleaned-artifact cache key. Bump on any
behavior change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yt2md.models import Transcript

CLEANER_VERSION = 1

# Derived from PodcastFillers (Zhu et al. 2022). Covers ~96% of annotated fillers
# in podcast audio. Excludes "mm"/"mhm" (agreement sounds, non-filler) and
# "you know"/"like"/"I mean" (context-dependent; removal risks meaning loss).
HARD_FILLERS = frozenset({"uh", "um", "uhm", "er", "ah"})


def clean(transcript: Transcript) -> Transcript:
    """Pure function: returns a cleaned copy of the transcript.

    No-op skeleton — concrete behavior added in subsequent tasks.
    """
    return transcript
