"""faster-whisper local transcription backend.

Optional dependency: import guarded so the module loads without the [local] extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from yt2md.errors import ConfigError, TranscriptionError
from yt2md.models import Segment, Transcript, Word
from yt2md.vocab_hint import extract_hints, format_for_whisper

if TYPE_CHECKING:
    from pathlib import Path

    from yt2md.config import Config
    from yt2md.models import VideoMetadata


def _import_faster_whisper() -> Any:  # noqa: ANN401
    """Import faster_whisper.WhisperModel; isolated for mocking + clear error."""
    from faster_whisper import WhisperModel  # type: ignore  # noqa: PLC0415,PGH003

    return WhisperModel


def transcribe_local(
    audio: Path,
    metadata: VideoMetadata,
    *,
    cfg: Config,
) -> tuple[Transcript, dict[str, Any]]:
    """Transcribe `audio` using faster-whisper locally.

    Returns (normalized_transcript, raw response dict). No diarization.
    """
    try:
        whisper_model_cls = _import_faster_whisper()
    except ImportError as e:
        msg = "faster-whisper not installed. Install with: pip install yt2llm[local]"
        raise ConfigError(msg) from e

    initial_prompt = (
        format_for_whisper(extract_hints(metadata)) if cfg.use_transcription_hint else None
    )

    try:
        model = whisper_model_cls(cfg.local_whisper_model, compute_type="auto")
        segments_iter, info = model.transcribe(
            str(audio),
            word_timestamps=True,
            initial_prompt=initial_prompt,
        )
        segments_list = list(segments_iter)
    except Exception as e:
        msg = f"Local whisper transcribe failed: {e}"
        raise TranscriptionError(msg) from e

    model_id = f"faster-whisper-{cfg.local_whisper_model}"
    transcript = _normalize_local_response(segments_list, info, model_id=model_id)
    raw = _serialize_local_response(segments_list, info)
    return transcript, raw


def _normalize_local_response(
    segments_raw: list[Any],
    info: Any,  # noqa: ANN401
    *,
    model_id: str,
) -> Transcript:
    segments = [_normalize_segment(s) for s in segments_raw]
    return Transcript(
        language=str(getattr(info, "language", "en")),
        duration_s=float(getattr(info, "duration", 0.0)),
        backend="local_whisper",
        model_id=model_id,
        chunked=False,
        segments=segments,
        speakers=[],
    )


def _normalize_segment(seg: Any) -> Segment:  # noqa: ANN401
    words: list[Word] = [
        Word(
            text=str(getattr(w, "word", "")).strip(),
            start=float(getattr(w, "start", 0.0)),
            end=float(getattr(w, "end", 0.0)),
            speaker=None,
        )
        for w in (getattr(seg, "words", None) or [])
    ]
    return Segment(
        start=float(seg.start),
        end=float(seg.end),
        text=str(seg.text).strip(),
        speaker=None,
        words=words,
    )


def _serialize_local_response(segments: list[Any], info: Any) -> dict[str, Any]:  # noqa: ANN401
    return {
        "language": getattr(info, "language", "en"),
        "duration": getattr(info, "duration", 0.0),
        "segments": [
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": str(s.text),
                "words": [
                    {
                        "word": str(getattr(w, "word", "")),
                        "start": float(w.start),
                        "end": float(w.end),
                    }
                    for w in (getattr(s, "words", None) or [])
                ],
            }
            for s in segments
        ],
    }
