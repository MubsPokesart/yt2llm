"""Compress stage: re-encode source audio to canonical Opus 32kbps mono via ffmpeg."""

from __future__ import annotations

import subprocess  # noqa: S404
from typing import TYPE_CHECKING

from yt2md.errors import TranscriptionError

if TYPE_CHECKING:
    from pathlib import Path

    from yt2md.config import Config


def compress(*, source: Path, destination: Path, cfg: Config) -> None:
    """Re-encode `source` to `destination` as Opus mono at cfg.audio_bitrate_kbps.

    Uses libopus with `voip` application tuning, optimized for speech.
    Raises TranscriptionError on ffmpeg failure.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i",
        str(source),
        "-vn",  # no video
        "-ac",
        "1",  # mono
        "-c:a",
        "libopus",
        "-b:a",
        f"{cfg.audio_bitrate_kbps}k",
        "-application",
        "voip",
        str(destination),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        msg = f"ffmpeg failed (exit {e.returncode}): {e.stderr}"
        raise TranscriptionError(msg) from e
