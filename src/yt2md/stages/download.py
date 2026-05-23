"""Download stage: yt-dlp wrapper.

Splits naturally into:
  - normalize_metadata(info_dict) → VideoMetadata + raises typed errors on bad video states
  - _map_ytdlp_error(exception) → typed YT2MDError subclass
  - download(url, cfg) → (source_audio_path, VideoMetadata, raw_info_dict)

This file holds the adapter (normalize_metadata). The yt-dlp call lives in the
download() function added by a later task; this task is testable without network.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from yt2md.errors import DownloadError, LivestreamNotEndedError, VideoUnavailableError, YT2MDError
from yt2md.models import Chapter, VideoMetadata

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
