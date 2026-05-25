"""Preflight check for the ffmpeg / ffprobe binaries.

Both `compress` and `chunk` shell out to these tools via subprocess. When the
binary is missing from PATH, Python raises `FileNotFoundError` from `subprocess.run`
which bubbles as an opaque OSError. Translate that to a typed ConfigError with
an install hint, surfaced at stage entry instead of from a low-level traceback.
"""

from __future__ import annotations

import shutil
from typing import Literal

from yt2md.errors import ConfigError

FfmpegTool = Literal["ffmpeg", "ffprobe"]

_INSTALL_HINT = (
    "Install ffmpeg (ships with ffprobe). "
    "macOS: brew install ffmpeg | Windows: winget install ffmpeg or scoop install ffmpeg | "
    "Linux: apt/dnf install ffmpeg. Docs: https://ffmpeg.org/download.html"
)


def require_ffmpeg_tool(name: FfmpegTool) -> None:
    """Raise ConfigError if `name` is not on PATH."""
    if shutil.which(name) is None:
        msg = f"{name} not found on PATH. {_INSTALL_HINT}"
        raise ConfigError(msg)
