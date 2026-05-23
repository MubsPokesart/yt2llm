"""OpenAI gpt-4o-transcribe(-diarize) backend.

Public surface:
  - transcribe_openai(audio, metadata, cfg) → Transcript + raw response dict (added in D.2)
  - normalize_openai_response(raw, model_id) → Transcript

Adapter is split out for unit testing without network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from yt2md.errors import TranscriptionError
from yt2md.models import Segment, Transcript, VideoMetadata, Word
from yt2md.vocab_hint import extract_hints, format_for_openai

if TYPE_CHECKING:
    from pathlib import Path

    from yt2md.config import Config


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


@retry(
    retry=retry_if_exception_type((
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
    )),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _call_openai_transcribe(
    client: OpenAI,
    audio_path: Path,
    model: str,
    prompt: str | None,
) -> object:
    kwargs: dict[str, Any] = {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities": ["word", "segment"],
    }
    if prompt:
        kwargs["prompt"] = prompt
    with audio_path.open("rb") as f:
        kwargs["file"] = f
        return client.audio.transcriptions.create(**kwargs)


def transcribe_openai(
    audio: Path,
    metadata: VideoMetadata,
    *,
    cfg: Config,
) -> tuple[Transcript, dict[str, Any]]:
    """Transcribe `audio` with gpt-4o-transcribe.

    Returns (normalized_transcript, raw_response_dict).
    """
    if cfg.openai_api_key is None:
        msg = "OPENAI_API_KEY not set"
        raise TranscriptionError(msg)

    client = OpenAI(api_key=cfg.openai_api_key.get_secret_value())
    prompt = format_for_openai(extract_hints(metadata)) if cfg.use_transcription_hint else None

    try:
        response = _call_openai_transcribe(client, audio, cfg.transcription_model, prompt)
    except Exception as e:
        msg = f"OpenAI transcribe failed: {e}"
        raise TranscriptionError(msg) from e

    raw: dict[str, Any] = (
        response.model_dump() if hasattr(response, "model_dump") else dict(response)  # type: ignore[call-overload]
    )
    transcript = normalize_openai_response(raw, model_id=cfg.transcription_model)
    return transcript, raw
