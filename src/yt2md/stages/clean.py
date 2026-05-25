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

CLEANER_VERSION = 2

# Derived from PodcastFillers (Zhu et al. 2022). Covers ~96% of annotated fillers
# in podcast audio. Excludes "mm"/"mhm" (agreement sounds, non-filler) and
# "you know"/"like"/"I mean" (context-dependent; removal risks meaning loss).
HARD_FILLERS = frozenset({"uh", "um", "uhm", "er", "ah"})

COLLAPSE_THRESHOLD = 0.95
NOISE_THRESHOLD = 0.01


def clean(transcript: Transcript) -> Transcript:
    """Pure function: returns a cleaned copy of the transcript.

    Steps:
      1. Drop HARD_FILLERS words from every segment; drop empty segments.
      2. Compute per-speaker duration (sum of word.end - word.start).
      3. Drop segments whose speaker contributes <1% of total duration (noise).
      4. If the dominant speaker holds >=95% of remaining duration, collapse:
         rewrite all word.speaker and segment.speaker to the dominant label,
         set transcript.speakers = [dominant].
    """
    filler_dropped = [_clean_segment(s) for s in transcript.segments]
    surviving = [s for s in filler_dropped if s is not None]

    if not transcript.speakers:
        return _replace_segments(transcript, surviving)

    per_speaker = _per_speaker_duration(surviving)
    total = sum(per_speaker.values())
    if total == 0:
        return _replace_segments(transcript, surviving)

    above_noise = {sp: d for sp, d in per_speaker.items() if d / total >= NOISE_THRESHOLD}
    surviving = [s for s in surviving if s.speaker in above_noise]

    if not surviving:
        return _replace_segments(transcript, surviving, speakers=[])

    dominant = max(above_noise, key=lambda sp: above_noise[sp])
    new_total = sum(above_noise.values())
    if above_noise[dominant] / new_total >= COLLAPSE_THRESHOLD:
        surviving = [_relabel_segment(s, dominant) for s in surviving]
        return _replace_segments(transcript, surviving, speakers=[dominant])

    return _replace_segments(
        transcript,
        surviving,
        speakers=sorted(above_noise.keys()),
    )


def _clean_segment(segment: Segment) -> Segment | None:
    if segment.words:
        kept: list[Word] = [w for w in segment.words if not _is_filler(w.text)]
        if not kept:
            return None
        return Segment(
            start=segment.start,
            end=segment.end,
            text=" ".join(w.text for w in kept),
            speaker=segment.speaker,
            words=kept,
        )
    # No word-level data: filter filler tokens from segment.text directly.
    # Preserves the segment when a backend (or a transcript shape) gives us
    # text without per-word timestamps.
    kept_tokens = [t for t in segment.text.split() if not _is_filler(t)]
    if not kept_tokens:
        return None
    return Segment(
        start=segment.start,
        end=segment.end,
        text=" ".join(kept_tokens),
        speaker=segment.speaker,
        words=[],
    )


def _is_filler(token: str) -> bool:
    normalized = token.lower().strip(string.punctuation + string.whitespace)
    return normalized in HARD_FILLERS


def _per_speaker_duration(segments: list[Segment]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for s in segments:
        for w in s.words:
            if w.speaker is None:
                continue
            durations[w.speaker] = durations.get(w.speaker, 0.0) + (w.end - w.start)
    return durations


def _relabel_segment(segment: Segment, label: str) -> Segment:
    relabeled_words = [
        Word(text=w.text, start=w.start, end=w.end, speaker=label) for w in segment.words
    ]
    return Segment(
        start=segment.start,
        end=segment.end,
        text=segment.text,
        speaker=label,
        words=relabeled_words,
    )


def _replace_segments(
    transcript: Transcript,
    segments: list[Segment],
    *,
    speakers: list[str] | None = None,
) -> Transcript:
    return Transcript(
        language=transcript.language,
        duration_s=transcript.duration_s,
        backend=transcript.backend,
        model_id=transcript.model_id,
        chunked=transcript.chunked,
        segments=segments,
        speakers=speakers if speakers is not None else transcript.speakers,
    )
