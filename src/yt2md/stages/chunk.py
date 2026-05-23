"""Chunking for long audio: silence-boundary split + offset stitching.

Public surface:
  - needs_chunking(audio, backend, cfg) → bool  (added in F.2)
  - split_at_silence(audio, backend, cfg) → list[Chunk]  (added in F.2)
  - stitch_transcripts(chunk_transcripts, offsets_s) → Transcript

Chunking is conditional. Most podcasts fit one request; only very long content
hits the split path.
"""

from __future__ import annotations

from yt2md.models import Segment, Transcript, Word


def stitch_transcripts(
    chunk_transcripts: list[Transcript],
    *,
    offsets_s: list[float],
) -> Transcript:
    """Concatenate per-chunk transcripts, applying each chunk's start offset to all timestamps.

    The result has `chunked=True` so the structurer prompt can soften speaker attribution.
    """
    if len(chunk_transcripts) != len(offsets_s):
        msg = "chunk_transcripts and offsets_s must have equal length"
        raise ValueError(msg)

    all_segments: list[Segment] = []
    max_end = 0.0
    for t, offset in zip(chunk_transcripts, offsets_s, strict=True):
        for seg in t.segments:
            shifted = _shift_segment(seg, offset)
            all_segments.append(shifted)
            max_end = max(max_end, shifted.end)

    first = chunk_transcripts[0]
    return Transcript(
        language=first.language,
        duration_s=max_end,
        backend=first.backend,
        model_id=first.model_id,
        chunked=True,
        segments=all_segments,
        speakers=_combined_speakers(chunk_transcripts),
    )


def _shift_segment(seg: Segment, offset: float) -> Segment:
    shifted_words = [
        Word(text=w.text, start=w.start + offset, end=w.end + offset, speaker=w.speaker)
        for w in seg.words
    ]
    return Segment(
        start=seg.start + offset,
        end=seg.end + offset,
        text=seg.text,
        speaker=seg.speaker,
        words=shifted_words,
    )


def _combined_speakers(transcripts: list[Transcript]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in transcripts:
        for sp in t.speakers:
            if sp not in seen:
                seen.add(sp)
                out.append(sp)
    return out
