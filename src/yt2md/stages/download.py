"""Download stage: yt-dlp wrapper.

Splits naturally into:
  - normalize_metadata(info_dict) → VideoMetadata + raises typed errors on bad video states
  - _map_ytdlp_error(exception) → typed YT2MDError subclass
  - download(url, cfg) → (source_audio_path, VideoMetadata, raw_info_dict)
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from yt_dlp import YoutubeDL  # type: ignore[import-untyped]
from yt_dlp.utils import DownloadError as YtdlpDownloadError  # type: ignore[import-untyped]

from yt2md.errors import DownloadError, LivestreamNotEndedError, VideoUnavailableError, YT2MDError
from yt2md.models import Chapter, VideoMetadata

if TYPE_CHECKING:
    from yt2md.config import Config

UPLOAD_DATE_LEN = 8


def normalize_metadata(info: dict[str, Any]) -> VideoMetadata:
    """Convert a yt-dlp info_dict into a VideoMetadata."""
    if info.get("is_live"):
        msg = "Video is a live stream that has not ended"
        raise LivestreamNotEndedError(msg)

    duration = float(info.get("duration", 0) or 0)
    if duration <= 0:
        msg = f"Video has zero duration; cannot transcribe (id={info.get('id')})"
        raise VideoUnavailableError(msg)

    upload_date_str = str(info.get("upload_date", ""))
    if len(upload_date_str) != UPLOAD_DATE_LEN or not upload_date_str.isdigit():
        msg = f"Invalid or missing upload_date: {upload_date_str!r}"
        raise VideoUnavailableError(msg)
    published = date(
        year=int(upload_date_str[:4]),
        month=int(upload_date_str[4:6]),
        day=int(upload_date_str[6:8]),
    )

    chapters = [
        Chapter(
            title=str(c["title"]),
            start_s=float(c["start_time"]),
            end_s=float(c["end_time"]),
        )
        for c in info.get("chapters") or []
    ]

    return VideoMetadata(
        video_id=str(info["id"]),
        url=str(info["webpage_url"]),
        title=str(info["title"]),
        channel=str(info["channel"]),
        channel_id=str(info["channel_id"]),
        published_date=published,
        duration_s=duration,
        description=str(info.get("description") or ""),
        chapters=chapters,
        tags=list(info.get("tags") or []),
        language=info.get("language"),
    )


def map_ytdlp_error(exc: Exception) -> YT2MDError:
    """Map a yt-dlp exception's message to a typed YT2MDError subclass.

    yt-dlp's exception hierarchy isn't fine-grained, so we match on substrings.
    yt-dlp version is pinned in pyproject.toml; bump intentionally and re-test on upgrade.
    """
    msg = str(exc).lower()

    cookie_hint = "Pass --cookies-from-browser firefox or --cookies cookies.txt."

    if "private video" in msg or "video unavailable" in msg or "has been removed" in msg:
        return VideoUnavailableError(str(exc))
    if "confirm your age" in msg or ("age" in msg and "restrict" in msg):
        return VideoUnavailableError(f"Age-restricted. {cookie_hint}")
    if "members-only" in msg or "members only" in msg or "join this channel" in msg:
        return VideoUnavailableError(f"Members-only. {cookie_hint}")
    if "not made this video available in your country" in msg or "geo" in msg:
        return VideoUnavailableError(str(exc))
    if "live event" in msg or "is live" in msg or "premiere" in msg:
        return LivestreamNotEndedError(str(exc))
    return DownloadError(str(exc))


_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/embed/|/shorts/)([\w-]+)")


def _extract_video_id(url: str) -> str:
    """Extract the 11-char YouTube video id from a URL."""
    match = _VIDEO_ID_RE.search(url)
    if match is None:
        msg = f"Could not extract video_id from URL: {url}"
        raise VideoUnavailableError(msg)
    return match.group(1)


def download(url: str, *, cfg: Config) -> tuple[Path, VideoMetadata, dict[str, Any]]:
    """Download audio + metadata via yt-dlp.

    Returns:
        (source_audio_path, normalized_metadata, raw_info_dict)
    """
    cache_dir = cfg.cache_dir
    video_id = _extract_video_id(url)
    out_dir = cache_dir / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "source_audio.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    if cfg.cookies_from_browser:
        opts["cookiesfrombrowser"] = (cfg.cookies_from_browser,)
    if cfg.cookies_file:
        opts["cookiefile"] = str(cfg.cookies_file)

    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = Path(ydl.prepare_filename(info))
    except YtdlpDownloadError as e:
        raise map_ytdlp_error(e) from e

    metadata = normalize_metadata(info)
    return filename, metadata, info
