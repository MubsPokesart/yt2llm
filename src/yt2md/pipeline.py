"""Pipeline orchestrator — the only module that knows the order of stages.

run(url, cfg) is the public API. cli.py calls it; tests call it; a future
web wrapper would call it. Stages don't know about each other.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from yt2md.cache import ArtifactPaths, cached, fingerprint
from yt2md.models import StructuredDoc, Transcript, VideoMetadata
from yt2md.stages.clean import CLEANER_VERSION, clean
from yt2md.stages.compress import compress
from yt2md.stages.download import _extract_video_id, download
from yt2md.stages.render import render
from yt2md.stages.structure import PROMPT_VERSION, structure
from yt2md.stages.transcribe import transcribe
from yt2md.stages.write import write
from yt2md.vocab_hint import VOCAB_HINT_VERSION

if TYPE_CHECKING:
    from pathlib import Path

    from yt2md.config import Config


# ---------------------------------------------------------------------------
# Dump helpers — Path.write_text returns int; mypy strict requires None.
# ---------------------------------------------------------------------------


def _dump_transcript(t: Transcript, p: Path) -> None:
    p.write_text(t.model_dump_json(), encoding="utf-8")


def _dump_structured_doc(d: StructuredDoc, p: Path) -> None:
    p.write_text(d.model_dump_json(), encoding="utf-8")


def run(url: str, *, cfg: Config) -> Path:
    """Execute the full pipeline. Returns the path to the written markdown."""
    metadata = _download_and_cache_metadata(url, cfg)
    paths = ArtifactPaths(cache_dir=cfg.cache_dir, video_id=metadata.video_id)

    audio = _compress_audio(url, paths, cfg)
    transcript = _transcribe_audio(audio, metadata, paths, cfg)
    cleaned = _clean_transcript(transcript, paths)
    doc = _structure_doc(cleaned, metadata, paths, cfg)
    markdown = render(doc, cleaned)
    return write(markdown=markdown, doc=doc, output_dir=cfg.output_dir)


def _download_and_cache_metadata(url: str, cfg: Config) -> VideoMetadata:
    """Cache metadata by video_id; only call yt-dlp if metadata.json missing."""
    video_id = _extract_video_id(url)
    paths = ArtifactPaths(cache_dir=cfg.cache_dir, video_id=video_id)

    if paths.metadata.exists():
        return VideoMetadata.model_validate_json(paths.metadata.read_text(encoding="utf-8"))

    _, metadata, raw = download(url, cfg=cfg)
    paths.metadata.parent.mkdir(parents=True, exist_ok=True)
    paths.metadata.write_text(metadata.model_dump_json(), encoding="utf-8")
    paths.metadata_raw.write_text(json.dumps(raw), encoding="utf-8")
    return metadata


def _compress_audio(url: str, paths: ArtifactPaths, cfg: Config) -> Path:
    """Produce compressed audio path, downloading source if needed."""
    compression_hash = fingerprint(cfg.audio_bitrate_kbps, cfg.audio_codec, "mono")
    audio_path = paths.audio(compression_hash=compression_hash)

    if audio_path.exists():
        return audio_path

    source = next(iter(paths.root.glob("source_audio.*")), None)
    if source is None:
        source, _, _ = download(url, cfg=cfg)

    compress(source=source, destination=audio_path, cfg=cfg)
    return audio_path


def _transcribe_audio(
    audio: Path,
    metadata: VideoMetadata,
    paths: ArtifactPaths,
    cfg: Config,
) -> Transcript:
    key = fingerprint(
        audio.stat().st_size,
        cfg.transcription_backend,
        cfg.transcription_model,
        cfg.local_whisper_model,
        cfg.use_transcription_hint,
        VOCAB_HINT_VERSION,
    )
    target = paths.transcript(input_hash=key)
    raw_target = paths.transcript_raw(input_hash=key)

    def _produce() -> Transcript:
        transcript, raws = transcribe(audio, metadata, cfg=cfg)
        raw_target.parent.mkdir(parents=True, exist_ok=True)
        raw_target.write_text(json.dumps(raws), encoding="utf-8")
        return transcript

    return cached(
        path=target,
        produce=_produce,
        load=lambda p: Transcript.model_validate_json(p.read_text(encoding="utf-8")),
        dump=_dump_transcript,
    )


def _clean_transcript(transcript: Transcript, paths: ArtifactPaths) -> Transcript:
    key = fingerprint(transcript.model_dump_json(), CLEANER_VERSION)
    target = paths.cleaned(input_hash=key)
    return cached(
        path=target,
        produce=lambda: clean(transcript),
        load=lambda p: Transcript.model_validate_json(p.read_text(encoding="utf-8")),
        dump=_dump_transcript,
    )


def _structure_doc(
    cleaned: Transcript,
    metadata: VideoMetadata,
    paths: ArtifactPaths,
    cfg: Config,
) -> StructuredDoc:
    key = fingerprint(cleaned.model_dump_json(), PROMPT_VERSION, cfg.structuring_model)
    target = paths.structured(input_hash=key)
    return cached(
        path=target,
        produce=lambda: structure(cleaned, metadata, cfg=cfg),
        load=lambda p: StructuredDoc.model_validate_json(p.read_text(encoding="utf-8")),
        dump=_dump_structured_doc,
    )
