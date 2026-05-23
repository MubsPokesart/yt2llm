"""OpenAI gpt-4o-transcribe(-diarize) backend.

Public surface:
  - transcribe_openai(audio, metadata, cfg) → Transcript + raw response dict (added in D.2)
  - normalize_openai_response(raw, model_id) → Transcript

Adapter is split out for unit testing without network.
"""

from __future__ import annotations

from typing import Any

from yt2md.models import Segment, Transcript, Word


def normalize_openai_response(raw: dict[str, Any], *, model_id: str) -> Transcript:
    """Convert OpenAI verbose_json transcribe response to our Transcript model.

    Strips leading whitespace from word.text (OpenAI emits leading spaces).
    Collects all distinct speakers from word/segment labels.
    """
    segments_raw = raw.get("segments") or []
    segments: list[Segment] = [_normalize_segment(s) for s in segments_raw]
    speakers = _collect_speakers(segments)
    return Transcript(
        language=str(raw.get("language", "en")),
        duration_s=float(raw.get("duration", 0.0)),
        backend="openai_transcribe",
        model_id=model_id,
        chunked=False,
        segments=segments,
        speakers=speakers,
    )


def _normalize_segment(raw_seg: dict[str, Any]) -> Segment:
    words = [_normalize_word(w) for w in raw_seg.get("words") or []]
    return Segment(
        start=float(raw_seg["start"]),
        end=float(raw_seg["end"]),
        text=str(raw_seg.get("text", "")).strip(),
        speaker=raw_seg.get("speaker"),
        words=words,
    )


def _normalize_word(raw_word: dict[str, Any]) -> Word:
    return Word(
        text=str(raw_word["word"]).strip(),
        start=float(raw_word["start"]),
        end=float(raw_word["end"]),
        speaker=raw_word.get("speaker"),
    )


def _collect_speakers(segments: list[Segment]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in segments:
        if s.speaker and s.speaker not in seen:
            seen.add(s.speaker)
            out.append(s.speaker)
    return out
