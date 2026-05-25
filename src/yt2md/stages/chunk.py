"""Chunking for long audio: silence-boundary split + offset stitching.

Public surface:
  - needs_chunking(audio, backend, cfg) → bool  (added in F.2)
  - split_at_silence(audio, backend, cfg) → list[Chunk]  (added in F.2)
  - stitch_transcripts(chunk_transcripts, offsets_s) → Transcript

Chunking is conditional. Most podcasts fit one request; only very long content
hits the split path.
"""

from __future__ import annotations

import subprocess  # noqa: S404
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from yt2md.config import Config

from yt2md.ffmpeg_preflight import require_ffmpeg_tool
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


# Hard caps per backend.
SIZE_CAP_MB = {
    "openai_transcribe": 20,
    "local_whisper": 200,
}
DURATION_CAP_S = {
    "openai_transcribe": 1500.0,  # 25 min
    "local_whisper": 3600.0,  # 1 hr
}

_DEFAULT_SIZE_CAP_MB = 200
_DEFAULT_DURATION_CAP_S = 3600.0

SILENCE_SEARCH_WINDOW_S = 30.0
SILENCE_MIN_S = 0.5
SILENCE_NOISE_DB = -30
_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class Chunk:
    path: Path
    start_offset_s: float
    duration_s: float


def needs_chunking(audio: Path, *, backend: str, cfg: Config) -> bool:  # noqa: ARG001
    """Decide if chunking is required for this backend."""
    size_mb = audio.stat().st_size / _BYTES_PER_MB
    if size_mb > SIZE_CAP_MB.get(backend, _DEFAULT_SIZE_CAP_MB):
        return True
    require_ffmpeg_tool("ffprobe")
    duration = _ffprobe_duration(audio)
    return duration > DURATION_CAP_S.get(backend, _DEFAULT_DURATION_CAP_S)


def split_at_silence(audio: Path, *, backend: str, cfg: Config) -> list[Chunk]:  # noqa: ARG001
    """Split `audio` into ~80%-of-cap chunks at silence boundaries."""
    require_ffmpeg_tool("ffmpeg")
    require_ffmpeg_tool("ffprobe")
    duration = _ffprobe_duration(audio)
    target_chunk_s = DURATION_CAP_S[backend] * 0.8
    num_chunks = max(
        1,
        int(duration / target_chunk_s) + (1 if duration % target_chunk_s else 0),
    )
    actual_chunk_s = duration / num_chunks

    ideal_cuts = [actual_chunk_s * i for i in range(1, num_chunks)]
    silences = _detect_silences(audio)
    boundaries = [_pick_nearest_silence(c, silences) for c in ideal_cuts]
    offsets = [0.0, *boundaries]
    durations = [
        offsets[i + 1] - offsets[i] if i < len(offsets) - 1 else duration - offsets[i]
        for i in range(len(offsets))
    ]

    chunks: list[Chunk] = []
    out_dir = audio.parent / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (offset, chunk_duration) in enumerate(zip(offsets, durations, strict=True)):
        chunk_path = out_dir / f"audio_{idx:02d}.opus"
        _cut_chunk(audio, chunk_path, start_s=offset, duration_s=chunk_duration)
        chunks.append(Chunk(path=chunk_path, start_offset_s=offset, duration_s=chunk_duration))
    return chunks


def _pick_nearest_silence(ideal_s: float, silences: list[float]) -> float:
    in_window = [s for s in silences if abs(s - ideal_s) <= SILENCE_SEARCH_WINDOW_S]
    if not in_window:
        return ideal_s
    return min(in_window, key=lambda s: abs(s - ideal_s))


def _ffprobe_duration(audio: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    return float(result.stdout.strip())


def _detect_silences(audio: Path) -> list[float]:
    """Run ffmpeg silencedetect; parse silence_start timestamps from stderr."""
    cmd = [
        "ffmpeg",
        "-i",
        str(audio),
        "-af",
        f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_S}",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)  # noqa: S603
    silences: list[float] = []
    for line in result.stderr.splitlines():
        if "silence_start:" in line:
            try:
                ts = float(line.split("silence_start:")[1].strip())
                silences.append(ts)
            except (IndexError, ValueError):
                continue
    return silences


def _cut_chunk(source: Path, destination: Path, *, start_s: float, duration_s: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_s),
        "-i",
        str(source),
        "-t",
        str(duration_s),
        "-c",
        "copy",
        str(destination),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
