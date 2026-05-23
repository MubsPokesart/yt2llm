# yt2llm Phase 3: API-Bound Stages — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Source spec:** `docs/superpowers/specs/2026-05-23-yt2llm-design.md`
**Index:** `docs/superpowers/plans/2026-05-23-yt2llm-index.md`
**Prereq:** Phase 2 complete (deterministic stages working with fixtures).

**Goal:** Implement the three external-API stages: `download` (yt-dlp), `compress` (ffmpeg), `transcribe` (dispatcher + OpenAI backend + faster-whisper backend + chunking), and `structure` (Gemini).

**Architecture:** Each stage wraps an external client at a thin adapter boundary. Tenacity retries at the SDK call site only. Mocked unit tests; live tests gated by `pytest -m live` + required API keys.

**Tech Stack:** Adds yt-dlp, openai SDK, google-genai SDK, faster-whisper (optional extra) to the Phase 1+2 stack.

**Definition of done:** All Phase 3 tasks checked off. `lint` + `typecheck` + `cover` all pass. Each stage has unit tests (mocked) and at least one live test (skipped without API keys).

---

## Non-negotiable discipline (recap)

Same as Phase 1+2. TDD red-green-refactor on every task. Lint+typecheck on every commit. 400 LOC ceiling. No abstractions without 3 concrete uses. Never `--no-verify`. See `docs/superpowers/plans/2026-05-23-yt2llm-index.md`.

Shorthand: `lint` = `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`. `typecheck` = `uv run mypy --strict src/`. `cover` = `uv run pytest tests/unit tests/integration --cov=src/yt2md --cov-report=term-missing --cov-fail-under=85`.

## Mocking discipline

External SDK calls are mocked **at the SDK boundary**, not at the stage boundary. That is: we mock `openai.OpenAI().audio.transcriptions.create(...)`, not our own `transcribe()`. Reason: testing the stage's adapter logic (input transformation, error mapping, retries) requires real shape inputs.

For mocks:
- Use `unittest.mock.patch` on the SDK's constructor or method
- Mock returns minimal shapes that match the real SDK response objects
- Never mock our own code unless it's a stub at a phase boundary (e.g., mock `pipeline.transcribe` from `pipeline.py`'s perspective in Phase 4)

For live tests:
- File location: `tests/live/`
- Marker: `@pytest.mark.live`
- Skipped automatically when API keys are missing (use a `skipif`)
- Use the smallest fixture audio that proves the stage works (30s)
- Cost per CI run if we ever flip them on: <$0.05

---

## Section A: download.py (yt-dlp wrapper)

---

### Task A.1: yt-dlp info dict → VideoMetadata adapter

**Files:**
- Create: `tests/unit/test_download_adapter.py`
- Create: `tests/fixtures/metadata/ytdlp_raw_sample.json`
- Create: `src/yt2md/stages/download.py`

- [ ] **Step 1: Capture a realistic yt-dlp info_dict fixture**

`tests/fixtures/metadata/ytdlp_raw_sample.json`:

```json
{
  "id": "abc123",
  "title": "Dopamine, Motivation & Drive",
  "channel": "Huberman Lab",
  "channel_id": "UCxxxx",
  "uploader": "Huberman Lab",
  "uploader_id": "@hubermanlab",
  "upload_date": "20240315",
  "duration": 5025,
  "description": "In this Huberman Lab episode, Andrew Huberman discusses dopamine.",
  "chapters": [
    {"title": "Introduction", "start_time": 0, "end_time": 60},
    {"title": "What dopamine actually does", "start_time": 60, "end_time": 2700}
  ],
  "tags": ["neuroscience", "dopamine"],
  "language": "en",
  "webpage_url": "https://www.youtube.com/watch?v=abc123",
  "is_live": false,
  "live_status": "not_live"
}
```

- [ ] **Step 2: Write the failing test**

`tests/unit/test_download_adapter.py`:

```python
"""Tests for yt-dlp info_dict → VideoMetadata adapter (no network)."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from yt2md.errors import LivestreamNotEndedError, VideoUnavailableError
from yt2md.models import VideoMetadata
from yt2md.stages.download import normalize_metadata


@pytest.fixture
def raw_info_dict(fixtures_dir: Path) -> dict[str, Any]:
    return json.loads(
        (fixtures_dir / "metadata" / "ytdlp_raw_sample.json").read_text(encoding="utf-8")
    )


class TestNormalizeMetadata:
    def test_extracts_basic_fields(self, raw_info_dict: dict[str, Any]) -> None:
        m = normalize_metadata(raw_info_dict)
        assert m.video_id == "abc123"
        assert m.title == "Dopamine, Motivation & Drive"
        assert m.channel == "Huberman Lab"
        assert m.channel_id == "UCxxxx"
        assert m.duration_s == 5025.0

    def test_upload_date_to_published_date(self, raw_info_dict: dict[str, Any]) -> None:
        m = normalize_metadata(raw_info_dict)
        assert m.published_date == date(2024, 3, 15)

    def test_chapters_mapped(self, raw_info_dict: dict[str, Any]) -> None:
        m = normalize_metadata(raw_info_dict)
        assert len(m.chapters) == 2
        assert m.chapters[0].title == "Introduction"
        assert m.chapters[0].start_s == 0.0
        assert m.chapters[0].end_s == 60.0

    def test_missing_chapters_yields_empty_list(self) -> None:
        m = normalize_metadata({
            "id": "x",
            "title": "T",
            "channel": "C",
            "channel_id": "UC",
            "upload_date": "20240101",
            "duration": 60,
            "description": "",
            "webpage_url": "https://www.youtube.com/watch?v=x",
        })
        assert m.chapters == []

    def test_missing_tags_yields_empty_list(self) -> None:
        m = normalize_metadata({
            "id": "x",
            "title": "T",
            "channel": "C",
            "channel_id": "UC",
            "upload_date": "20240101",
            "duration": 60,
            "description": "",
            "webpage_url": "https://www.youtube.com/watch?v=x",
        })
        assert m.tags == []

    def test_missing_language_is_none(self) -> None:
        m = normalize_metadata({
            "id": "x",
            "title": "T",
            "channel": "C",
            "channel_id": "UC",
            "upload_date": "20240101",
            "duration": 60,
            "description": "",
            "webpage_url": "https://www.youtube.com/watch?v=x",
        })
        assert m.language is None

    def test_returns_videometadata_instance(self, raw_info_dict: dict[str, Any]) -> None:
        m = normalize_metadata(raw_info_dict)
        assert isinstance(m, VideoMetadata)


class TestLiveDetection:
    def test_is_live_raises_livestream_error(self) -> None:
        with pytest.raises(LivestreamNotEndedError):
            normalize_metadata({"id": "x", "title": "T", "channel": "C", "channel_id": "UC",
                                "upload_date": "20240101", "duration": 0, "description": "",
                                "webpage_url": "https://www.youtube.com/watch?v=x",
                                "is_live": True})

    def test_live_status_post_live_ok(self, raw_info_dict: dict[str, Any]) -> None:
        # A finished livestream is fine — has full audio.
        raw_info_dict["live_status"] = "post_live"
        m = normalize_metadata(raw_info_dict)
        assert m.video_id == "abc123"


class TestNoAudio:
    def test_zero_duration_raises_unavailable(self) -> None:
        with pytest.raises(VideoUnavailableError):
            normalize_metadata({"id": "x", "title": "T", "channel": "C", "channel_id": "UC",
                                "upload_date": "20240101", "duration": 0, "description": "",
                                "webpage_url": "https://www.youtube.com/watch?v=x"})
```

- [ ] **Step 3: Run — confirm fails**

```bash
uv run pytest tests/unit/test_download_adapter.py -v
```

Expected: FAIL.

- [ ] **Step 4: Write `src/yt2md/stages/download.py` adapter only**

```python
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

from yt2md.errors import LivestreamNotEndedError, VideoUnavailableError
from yt2md.models import Chapter, VideoMetadata


def normalize_metadata(info: dict[str, Any]) -> VideoMetadata:
    """Convert a yt-dlp info_dict into a VideoMetadata.

    Raises:
        LivestreamNotEndedError: if the video is currently live.
        VideoUnavailableError: if the video has zero duration or other unrecoverable
            data missing.
    """
    if info.get("is_live"):
        msg = "Video is a live stream that has not ended"
        raise LivestreamNotEndedError(msg)

    duration = float(info.get("duration", 0) or 0)
    if duration <= 0:
        msg = f"Video has zero duration; cannot transcribe (id={info.get('id')})"
        raise VideoUnavailableError(msg)

    upload_date_str = str(info.get("upload_date", ""))
    if len(upload_date_str) != 8 or not upload_date_str.isdigit():
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
```

- [ ] **Step 5: Run — confirm passes**

```bash
uv run pytest tests/unit/test_download_adapter.py -v
```

Expected: 9 PASS.

- [ ] **Step 6: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/download.py tests/unit/test_download_adapter.py tests/fixtures/metadata/ytdlp_raw_sample.json
git commit -m "$(cat <<'EOF'
feat(download): yt-dlp info_dict → VideoMetadata adapter with live/no-audio detection

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task A.2: yt-dlp error message → typed exception mapping

**Files:**
- Create: `tests/unit/test_download_error_mapping.py`
- Modify: `src/yt2md/stages/download.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_download_error_mapping.py`:

```python
"""Tests for mapping yt-dlp DownloadError strings to typed exceptions."""

import pytest

from yt2md.errors import LivestreamNotEndedError, VideoUnavailableError, YT2MDError
from yt2md.stages.download import map_ytdlp_error


class TestErrorMapping:
    @pytest.mark.parametrize(
        "ytdlp_message",
        [
            "ERROR: [youtube] Private video. Sign in if you've been granted access",
            "ERROR: [youtube] Video unavailable",
            "ERROR: [youtube] This video has been removed",
        ],
    )
    def test_private_or_removed(self, ytdlp_message: str) -> None:
        err = map_ytdlp_error(Exception(ytdlp_message))
        assert isinstance(err, VideoUnavailableError)

    def test_age_restricted_includes_cookie_hint(self) -> None:
        err = map_ytdlp_error(Exception(
            "ERROR: [youtube] Sign in to confirm your age. This video may be inappropriate"
        ))
        assert isinstance(err, VideoUnavailableError)
        assert "cookies-from-browser" in str(err).lower()

    def test_members_only_includes_cookie_hint(self) -> None:
        err = map_ytdlp_error(Exception(
            "ERROR: [youtube] Join this channel to get access to members-only content"
        ))
        assert isinstance(err, VideoUnavailableError)
        assert "cookies" in str(err).lower()

    def test_geoblocked(self) -> None:
        err = map_ytdlp_error(Exception(
            "ERROR: [youtube] The uploader has not made this video available in your country"
        ))
        assert isinstance(err, VideoUnavailableError)

    def test_livestream_not_ended(self) -> None:
        err = map_ytdlp_error(Exception(
            "ERROR: This live event will begin in 2 hours"
        ))
        assert isinstance(err, LivestreamNotEndedError)

    def test_unmatched_falls_back_to_root(self) -> None:
        err = map_ytdlp_error(Exception("Some entirely unknown error string"))
        assert isinstance(err, YT2MDError)
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_download_error_mapping.py -v
```

Expected: FAIL — `map_ytdlp_error` does not exist.

- [ ] **Step 3: Append to `src/yt2md/stages/download.py`**

```python
from yt2md.errors import DownloadError


def map_ytdlp_error(exc: Exception) -> YT2MDError:
    """Map a yt-dlp exception's message to a typed YT2MDError subclass.

    yt-dlp's exception hierarchy isn't fine-grained, so we match on substrings.
    yt-dlp version is pinned in pyproject.toml; bump intentionally and re-test on upgrade.
    """
    msg = str(exc).lower()

    cookie_hint = "Pass --cookies-from-browser firefox or --cookies cookies.txt."

    if "private video" in msg or "video unavailable" in msg or "has been removed" in msg:
        return VideoUnavailableError(str(exc))
    if "confirm your age" in msg or "age" in msg and "restrict" in msg:
        return VideoUnavailableError(f"Age-restricted. {cookie_hint}")
    if "members-only" in msg or "members only" in msg or "join this channel" in msg:
        return VideoUnavailableError(f"Members-only. {cookie_hint}")
    if "not made this video available in your country" in msg or "geo" in msg:
        return VideoUnavailableError(str(exc))
    if "live event" in msg or "is live" in msg or "premiere" in msg:
        return LivestreamNotEndedError(str(exc))
    return DownloadError(str(exc))
```

Add to the existing imports if needed:

```python
from yt2md.errors import DownloadError, LivestreamNotEndedError, VideoUnavailableError, YT2MDError
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_download_error_mapping.py -v
```

Expected: 8 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/download.py tests/unit/test_download_error_mapping.py
git commit -m "$(cat <<'EOF'
feat(download): map_ytdlp_error() — string-match to typed exceptions

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task A.3: download() — yt-dlp call with mocked SDK

**Files:**
- Create: `tests/unit/test_download_call.py`
- Modify: `src/yt2md/stages/download.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_download_call.py`:

```python
"""Tests for download() — yt-dlp call mocked at the SDK boundary."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from yt2md.config import Config
from yt2md.errors import LivestreamNotEndedError, VideoUnavailableError
from yt2md.stages.download import download


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        google_api_key="g",  # type: ignore[arg-type]
        cache_dir=tmp_path,
        output_dir=tmp_path / "out",
    )


@pytest.fixture
def fake_info(fixtures_dir: Path) -> dict[str, Any]:
    return json.loads(
        (fixtures_dir / "metadata" / "ytdlp_raw_sample.json").read_text(encoding="utf-8")
    )


class TestDownloadHappyPath:
    def test_returns_audio_path_metadata_and_raw(
        self, cfg: Config, fake_info: dict[str, Any], tmp_path: Path
    ) -> None:
        # Simulate yt-dlp writing an audio file and producing info_dict.
        fake_audio = tmp_path / "abc123" / "source_audio.m4a"
        fake_audio.parent.mkdir(parents=True)
        fake_audio.write_bytes(b"\x00" * 100)

        fake_ydl = MagicMock()
        fake_ydl.__enter__ = lambda self: self
        fake_ydl.__exit__ = lambda self, *args: False
        fake_ydl.extract_info.return_value = fake_info
        fake_ydl.prepare_filename.return_value = str(fake_audio)

        with patch("yt2md.stages.download.YoutubeDL", return_value=fake_ydl):
            audio_path, metadata, raw = download(
                "https://www.youtube.com/watch?v=abc123",
                cfg=cfg,
            )

        assert audio_path == fake_audio
        assert metadata.video_id == "abc123"
        assert raw == fake_info


class TestDownloadErrorMapping:
    def test_yt_dlp_private_raises_video_unavailable(self, cfg: Config) -> None:
        from yt_dlp.utils import DownloadError as YtdlError

        fake_ydl = MagicMock()
        fake_ydl.__enter__ = lambda self: self
        fake_ydl.__exit__ = lambda self, *args: False
        fake_ydl.extract_info.side_effect = YtdlError("ERROR: [youtube] Private video")

        with patch("yt2md.stages.download.YoutubeDL", return_value=fake_ydl):
            with pytest.raises(VideoUnavailableError):
                download("https://www.youtube.com/watch?v=x", cfg=cfg)


class TestDownloadLivestream:
    def test_live_video_raises_livestream_error(
        self, cfg: Config, fake_info: dict[str, Any]
    ) -> None:
        fake_info["is_live"] = True
        fake_ydl = MagicMock()
        fake_ydl.__enter__ = lambda self: self
        fake_ydl.__exit__ = lambda self, *args: False
        fake_ydl.extract_info.return_value = fake_info

        with patch("yt2md.stages.download.YoutubeDL", return_value=fake_ydl):
            with pytest.raises(LivestreamNotEndedError):
                download("https://www.youtube.com/watch?v=x", cfg=cfg)


class TestCookiesPassthrough:
    def test_cookies_from_browser_in_yt_dlp_opts(self, cfg: Config) -> None:
        cfg_with_cookies = cfg.model_copy(update={"cookies_from_browser": "firefox"})
        fake_ydl = MagicMock()
        fake_ydl.__enter__ = lambda self: self
        fake_ydl.__exit__ = lambda self, *args: False
        fake_ydl.extract_info.return_value = {
            "id": "x", "title": "T", "channel": "C", "channel_id": "UC",
            "upload_date": "20240101", "duration": 60, "description": "",
            "webpage_url": "https://www.youtube.com/watch?v=x",
        }
        fake_ydl.prepare_filename.return_value = "/tmp/x.m4a"
        # Ensure the file exists for the assert path
        (Path("/tmp") / "x.m4a").write_bytes(b"x") if Path("/tmp").exists() else None

        with patch("yt2md.stages.download.YoutubeDL") as ydl_class:
            ydl_class.return_value = fake_ydl
            try:
                download("https://www.youtube.com/watch?v=x", cfg=cfg_with_cookies)
            except FileNotFoundError:
                pass  # OS-dependent /tmp behavior; the assert below is what matters
            opts = ydl_class.call_args[0][0]
            assert opts.get("cookiesfrombrowser") == ("firefox",)
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_download_call.py -v
```

Expected: FAIL — `download` does not exist; `YoutubeDL` not imported.

- [ ] **Step 3: Append to `src/yt2md/stages/download.py`**

```python
from pathlib import Path

from yt_dlp import YoutubeDL  # type: ignore[import-untyped]
from yt_dlp.utils import DownloadError as YtdlpDownloadError  # type: ignore[import-untyped]

from yt2md.config import Config


def download(url: str, *, cfg: Config) -> tuple[Path, VideoMetadata, dict[str, Any]]:
    """Download audio + metadata via yt-dlp.

    Returns:
        (source_audio_path, normalized_metadata, raw_info_dict)

    The source_audio_path is whatever format yt-dlp pulled (m4a, webm, opus, etc.).
    The compress stage converts it to the canonical Opus 32kbps mono. Both files
    are kept in cache; only the compressed one is the transcribe input.
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


_VIDEO_ID_RE = __import__("re").compile(r"(?:v=|youtu\.be/|/embed/|/shorts/)([\w-]{11})")


def _extract_video_id(url: str) -> str:
    """Extract the 11-char YouTube video id from a URL."""
    match = _VIDEO_ID_RE.search(url)
    if match is None:
        msg = f"Could not extract video_id from URL: {url}"
        raise VideoUnavailableError(msg)
    return match.group(1)
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_download_call.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/download.py tests/unit/test_download_call.py
git commit -m "$(cat <<'EOF'
feat(download): download() wraps yt-dlp with cookies, error mapping, video_id extraction

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task A.4: Live test for download (skipped without network)

**Files:**
- Create: `tests/live/test_live_download.py`

A real download of a tiny, stable public YouTube video. Skipped by default; runs only with `pytest -m live`.

- [ ] **Step 1: Write the live test**

`tests/live/test_live_download.py`:

```python
"""Live test for download() — hits real YouTube. Skipped without `-m live`."""

from pathlib import Path

import pytest

from yt2md.config import Config
from yt2md.stages.download import download

# A short, permanently-public, copyright-free test video (10 seconds).
# If this URL becomes unavailable, replace with another stable public domain video.
STABLE_TEST_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # "Me at the zoo" (19s, first YouTube video)


@pytest.mark.live
def test_live_download_real_youtube_video(tmp_path: Path) -> None:
    cfg = Config(google_api_key="g", cache_dir=tmp_path, output_dir=tmp_path / "out")  # type: ignore[arg-type]
    audio_path, meta, raw = download(STABLE_TEST_URL, cfg=cfg)
    assert audio_path.exists()
    assert audio_path.stat().st_size > 1000  # at least a few KB
    assert meta.duration_s > 5  # 19s video
    assert meta.video_id == "jNQXAC9IVRw"
    assert isinstance(raw, dict)
```

- [ ] **Step 2: Run with `-m live` if you have network**

```bash
uv run pytest tests/live/test_live_download.py -v -m live
```

Expected: PASS if network + yt-dlp work. SKIPPED if `-m live` not passed.

- [ ] **Step 3: Lint + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add tests/live/test_live_download.py
git commit -m "$(cat <<'EOF'
test(download): live test against stable public YouTube video

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section B: compress.py (ffmpeg)

---

### Task B.1: compress() invokes ffmpeg with canonical args

**Files:**
- Create: `tests/unit/test_compress.py`
- Create: `src/yt2md/stages/compress.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_compress.py`:

```python
"""Tests for compress() — ffmpeg subprocess mocked."""

from pathlib import Path
from unittest.mock import patch

import pytest

from yt2md.config import Config
from yt2md.errors import TranscriptionError
from yt2md.stages.compress import compress


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        google_api_key="g",  # type: ignore[arg-type]
        cache_dir=tmp_path,
        output_dir=tmp_path / "out",
        audio_bitrate_kbps=32,
    )


class TestCompress:
    def test_invokes_ffmpeg_with_canonical_args(self, cfg: Config, tmp_path: Path) -> None:
        src = tmp_path / "source.m4a"
        src.write_bytes(b"x")
        out = tmp_path / "out.opus"

        with patch("yt2md.stages.compress.subprocess.run") as run:
            run.return_value.returncode = 0
            compress(source=src, destination=out, cfg=cfg)
            args = run.call_args[0][0]

        assert args[0] == "ffmpeg"
        assert "-i" in args
        assert str(src) in args
        assert str(out) in args
        assert "-vn" in args  # no video
        assert "-ac" in args
        assert "1" in args  # mono
        assert "-c:a" in args
        assert "libopus" in args
        assert "-b:a" in args
        assert "32k" in args

    def test_creates_output_directory(self, cfg: Config, tmp_path: Path) -> None:
        src = tmp_path / "source.m4a"
        src.write_bytes(b"x")
        out = tmp_path / "nested" / "out.opus"

        with patch("yt2md.stages.compress.subprocess.run") as run:
            run.return_value.returncode = 0
            compress(source=src, destination=out, cfg=cfg)

        assert out.parent.exists()

    def test_ffmpeg_failure_raises_typed(self, cfg: Config, tmp_path: Path) -> None:
        import subprocess

        src = tmp_path / "source.m4a"
        src.write_bytes(b"x")
        out = tmp_path / "out.opus"

        with patch("yt2md.stages.compress.subprocess.run") as run:
            run.side_effect = subprocess.CalledProcessError(1, ["ffmpeg"], stderr="boom")
            with pytest.raises(TranscriptionError, match="ffmpeg"):
                compress(source=src, destination=out, cfg=cfg)
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_compress.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write `src/yt2md/stages/compress.py`**

```python
"""Compress stage: re-encode source audio to canonical Opus 32kbps mono via ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from yt2md.config import Config
from yt2md.errors import TranscriptionError


def compress(*, source: Path, destination: Path, cfg: Config) -> None:
    """Re-encode `source` to `destination` as Opus mono at cfg.audio_bitrate_kbps.

    Uses libopus with `voip` application tuning, optimized for speech.
    Raises TranscriptionError on ffmpeg failure (typed as such because the next
    stage that fails after this would be transcribe).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-i", str(source),
        "-vn",  # no video
        "-ac", "1",  # mono
        "-c:a", "libopus",
        "-b:a", f"{cfg.audio_bitrate_kbps}k",
        "-application", "voip",
        str(destination),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    except subprocess.CalledProcessError as e:
        msg = f"ffmpeg failed (exit {e.returncode}): {e.stderr}"
        raise TranscriptionError(msg) from e
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_compress.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/compress.py tests/unit/test_compress.py
git commit -m "$(cat <<'EOF'
feat(compress): ffmpeg subprocess wrapper producing canonical Opus 32kbps mono

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section C: transcribe — backend dispatcher

---

### Task C.1: resolve_backend() — auto-fallback logic

**Files:**
- Create: `tests/unit/test_transcribe_dispatch.py`
- Create: `src/yt2md/stages/transcribe.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_transcribe_dispatch.py`:

```python
"""Tests for resolve_backend() — auto, explicit, and fallback semantics."""

from pathlib import Path
from unittest.mock import patch

import pytest

from yt2md.config import Config
from yt2md.errors import ConfigError
from yt2md.stages.transcribe import resolve_backend


def _cfg(**kwargs: object) -> Config:
    defaults: dict[str, object] = {"google_api_key": "g"}
    defaults.update(kwargs)
    return Config(**defaults)  # type: ignore[arg-type]


class TestExplicit:
    def test_explicit_openai_requires_api_key(self) -> None:
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            resolve_backend(_cfg(transcription_backend="openai_transcribe", openai_api_key=None))

    def test_explicit_openai_succeeds_with_key(self) -> None:
        cfg = _cfg(transcription_backend="openai_transcribe", openai_api_key="key")
        assert resolve_backend(cfg) == "openai_transcribe"

    def test_explicit_local_requires_faster_whisper(self) -> None:
        cfg = _cfg(transcription_backend="local_whisper")
        with patch("yt2md.stages.transcribe._faster_whisper_installed", return_value=False):
            with pytest.raises(ConfigError, match="faster-whisper"):
                resolve_backend(cfg)

    def test_explicit_local_succeeds_when_installed(self) -> None:
        cfg = _cfg(transcription_backend="local_whisper")
        with patch("yt2md.stages.transcribe._faster_whisper_installed", return_value=True):
            assert resolve_backend(cfg) == "local_whisper"


class TestAuto:
    def test_auto_picks_openai_when_key_present(self) -> None:
        cfg = _cfg(transcription_backend="auto", openai_api_key="key")
        assert resolve_backend(cfg) == "openai_transcribe"

    def test_auto_falls_back_to_local_when_key_missing(self) -> None:
        cfg = _cfg(transcription_backend="auto", openai_api_key=None)
        with patch("yt2md.stages.transcribe._faster_whisper_installed", return_value=True):
            assert resolve_backend(cfg) == "local_whisper"

    def test_auto_hard_error_when_neither_available(self) -> None:
        cfg = _cfg(transcription_backend="auto", openai_api_key=None)
        with patch("yt2md.stages.transcribe._faster_whisper_installed", return_value=False):
            with pytest.raises(ConfigError, match="No transcription backend"):
                resolve_backend(cfg)
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_transcribe_dispatch.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write `src/yt2md/stages/transcribe.py`**

```python
"""Transcribe stage: backend dispatcher.

Backends live in stages/transcribe_backends/. This module:
  - resolves which backend to use given Config (auto / explicit)
  - dispatches to the chosen backend
  - is the place chunking integrates (added in a later task)
"""

from __future__ import annotations

import importlib.util
from typing import Literal

from yt2md.config import Config
from yt2md.errors import ConfigError

ResolvedBackend = Literal["openai_transcribe", "local_whisper"]


def resolve_backend(cfg: Config) -> ResolvedBackend:
    """Resolve the transcription backend based on Config and installed packages.

    Raises ConfigError if the configured (or auto-resolved) backend is unavailable.
    """
    choice = cfg.transcription_backend

    if choice == "openai_transcribe":
        if cfg.openai_api_key is None:
            msg = "transcription_backend=openai_transcribe requires OPENAI_API_KEY to be set"
            raise ConfigError(msg)
        return "openai_transcribe"

    if choice == "local_whisper":
        if not _faster_whisper_installed():
            msg = (
                "transcription_backend=local_whisper requires the [local] extra. "
                "Install with: pip install yt2llm[local]"
            )
            raise ConfigError(msg)
        return "local_whisper"

    # choice == "auto"
    if cfg.openai_api_key is not None:
        return "openai_transcribe"
    if _faster_whisper_installed():
        return "local_whisper"
    msg = (
        "No transcription backend available. Set OPENAI_API_KEY or install "
        "the local extra: pip install yt2llm[local]"
    )
    raise ConfigError(msg)


def _faster_whisper_installed() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_transcribe_dispatch.py -v
```

Expected: 7 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/transcribe.py tests/unit/test_transcribe_dispatch.py
git commit -m "$(cat <<'EOF'
feat(transcribe): resolve_backend() with auto-fallback and explicit modes

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section D: transcribe_backends/openai.py

---

### Task D.1: OpenAI response → Transcript adapter

**Files:**
- Create: `tests/unit/test_openai_backend_adapter.py`
- Create: `tests/fixtures/transcripts/openai_raw_sample.json`
- Create: `src/yt2md/stages/transcribe_backends/__init__.py`
- Create: `src/yt2md/stages/transcribe_backends/openai.py`

- [ ] **Step 1: Capture a realistic OpenAI response fixture**

`tests/fixtures/transcripts/openai_raw_sample.json`:

```json
{
  "language": "en",
  "duration": 8.0,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 8.0,
      "text": " Hello world this is a test.",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": " Hello",  "start": 0.0,  "end": 0.5, "speaker": "SPEAKER_00"},
        {"word": " world",  "start": 0.6,  "end": 1.0, "speaker": "SPEAKER_00"},
        {"word": " this",   "start": 1.1,  "end": 1.4, "speaker": "SPEAKER_00"},
        {"word": " is",     "start": 1.5,  "end": 1.7, "speaker": "SPEAKER_00"},
        {"word": " a",      "start": 1.8,  "end": 1.9, "speaker": "SPEAKER_00"},
        {"word": " test.",  "start": 2.0,  "end": 8.0, "speaker": "SPEAKER_00"}
      ]
    }
  ],
  "text": "Hello world this is a test."
}
```

- [ ] **Step 2: Write the failing test**

`tests/unit/test_openai_backend_adapter.py`:

```python
"""Tests for the OpenAI response → Transcript adapter (no network)."""

import json
from pathlib import Path

import pytest

from yt2md.models import Transcript
from yt2md.stages.transcribe_backends.openai import normalize_openai_response


@pytest.fixture
def raw(fixtures_dir: Path) -> dict[str, object]:
    return json.loads(
        (fixtures_dir / "transcripts" / "openai_raw_sample.json").read_text(encoding="utf-8")
    )


class TestNormalizeOpenAIResponse:
    def test_returns_transcript(self, raw: dict[str, object]) -> None:
        t = normalize_openai_response(raw, model_id="gpt-4o-transcribe")
        assert isinstance(t, Transcript)

    def test_backend_field(self, raw: dict[str, object]) -> None:
        t = normalize_openai_response(raw, model_id="gpt-4o-transcribe")
        assert t.backend == "openai_transcribe"
        assert t.model_id == "gpt-4o-transcribe"

    def test_duration_mapped(self, raw: dict[str, object]) -> None:
        t = normalize_openai_response(raw, model_id="gpt-4o-transcribe")
        assert t.duration_s == 8.0

    def test_speakers_collected(self, raw: dict[str, object]) -> None:
        t = normalize_openai_response(raw, model_id="gpt-4o-transcribe")
        assert t.speakers == ["SPEAKER_00"]

    def test_word_timestamps_preserved(self, raw: dict[str, object]) -> None:
        t = normalize_openai_response(raw, model_id="gpt-4o-transcribe")
        first_word = t.segments[0].words[0]
        assert first_word.text == "Hello"  # leading space stripped
        assert first_word.start == 0.0
        assert first_word.end == 0.5
        assert first_word.speaker == "SPEAKER_00"

    def test_chunked_flag_false_by_default(self, raw: dict[str, object]) -> None:
        t = normalize_openai_response(raw, model_id="gpt-4o-transcribe")
        assert t.chunked is False

    def test_undiarized_response_yields_none_speakers(self) -> None:
        no_speaker = {
            "language": "en",
            "duration": 2.0,
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 2.0,
                    "text": " Hi.",
                    "words": [{"word": " Hi.", "start": 0.0, "end": 2.0}],
                }
            ],
            "text": "Hi.",
        }
        t = normalize_openai_response(no_speaker, model_id="gpt-4o-transcribe")
        assert t.speakers == []
        assert t.segments[0].speaker is None
        assert t.segments[0].words[0].speaker is None
```

- [ ] **Step 3: Run — confirm fails**

```bash
uv run pytest tests/unit/test_openai_backend_adapter.py -v
```

Expected: FAIL.

- [ ] **Step 4: Write the backends package + openai adapter**

`src/yt2md/stages/transcribe_backends/__init__.py`:

```python
"""Per-backend transcription implementations. Each module exposes one transcribe function."""
```

`src/yt2md/stages/transcribe_backends/openai.py`:

```python
"""OpenAI gpt-4o-transcribe(-diarize) backend.

Public surface:
  - transcribe_openai(audio, metadata, cfg) → Transcript + raw response dict
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
```

- [ ] **Step 5: Run — confirm passes**

```bash
uv run pytest tests/unit/test_openai_backend_adapter.py -v
```

Expected: 7 PASS.

- [ ] **Step 6: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/transcribe_backends/ tests/unit/test_openai_backend_adapter.py tests/fixtures/transcripts/openai_raw_sample.json
git commit -m "$(cat <<'EOF'
feat(transcribe-openai): adapter from OpenAI verbose_json to Transcript

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task D.2: transcribe_openai() — SDK call with vocab hint + retry

**Files:**
- Create: `tests/unit/test_openai_backend_call.py`
- Modify: `src/yt2md/stages/transcribe_backends/openai.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_openai_backend_call.py`:

```python
"""Tests for transcribe_openai() — OpenAI SDK call mocked."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from yt2md.config import Config
from yt2md.errors import TranscriptionError
from yt2md.stages.transcribe_backends.openai import transcribe_openai


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        google_api_key="g",  # type: ignore[arg-type]
        openai_api_key="okey",  # type: ignore[arg-type]
        cache_dir=tmp_path,
        output_dir=tmp_path / "out",
    )


@pytest.fixture
def fake_audio(tmp_path: Path) -> Path:
    p = tmp_path / "a.opus"
    p.write_bytes(b"\x00" * 100)
    return p


@pytest.fixture
def fake_metadata(huberman_metadata):  # type: ignore[no-untyped-def]
    return huberman_metadata


@pytest.fixture
def fake_response(fixtures_dir: Path) -> SimpleNamespace:
    raw = json.loads(
        (fixtures_dir / "transcripts" / "openai_raw_sample.json").read_text(encoding="utf-8")
    )
    # SDK returns an object with a .model_dump() method on newer versions; simulate that.
    obj = SimpleNamespace()
    obj.model_dump = lambda: raw
    return obj


class TestTranscribeOpenAI:
    def test_returns_transcript_and_raw(
        self, cfg: Config, fake_audio: Path, fake_metadata, fake_response: SimpleNamespace
    ) -> None:  # type: ignore[no-untyped-def]
        with patch("yt2md.stages.transcribe_backends.openai.OpenAI") as openai_cls:
            client = MagicMock()
            client.audio.transcriptions.create.return_value = fake_response
            openai_cls.return_value = client

            transcript, raw = transcribe_openai(fake_audio, fake_metadata, cfg=cfg)

        assert transcript.backend == "openai_transcribe"
        assert transcript.duration_s == 8.0
        assert raw == fake_response.model_dump()

    def test_passes_vocab_hint_as_prompt(
        self, cfg: Config, fake_audio: Path, fake_metadata, fake_response: SimpleNamespace
    ) -> None:  # type: ignore[no-untyped-def]
        with patch("yt2md.stages.transcribe_backends.openai.OpenAI") as openai_cls:
            client = MagicMock()
            client.audio.transcriptions.create.return_value = fake_response
            openai_cls.return_value = client

            transcribe_openai(fake_audio, fake_metadata, cfg=cfg)

            kwargs = client.audio.transcriptions.create.call_args.kwargs
            assert "prompt" in kwargs
            assert "Huberman" in kwargs["prompt"]

    def test_skipped_hint_when_disabled(
        self, cfg: Config, fake_audio: Path, fake_metadata, fake_response: SimpleNamespace
    ) -> None:  # type: ignore[no-untyped-def]
        cfg_no_hint = cfg.model_copy(update={"use_transcription_hint": False})
        with patch("yt2md.stages.transcribe_backends.openai.OpenAI") as openai_cls:
            client = MagicMock()
            client.audio.transcriptions.create.return_value = fake_response
            openai_cls.return_value = client

            transcribe_openai(fake_audio, fake_metadata, cfg=cfg_no_hint)

            kwargs = client.audio.transcriptions.create.call_args.kwargs
            assert kwargs.get("prompt") is None or kwargs.get("prompt") == ""

    def test_api_error_raises_typed(
        self, cfg: Config, fake_audio: Path, fake_metadata
    ) -> None:  # type: ignore[no-untyped-def]
        with patch("yt2md.stages.transcribe_backends.openai.OpenAI") as openai_cls:
            client = MagicMock()
            client.audio.transcriptions.create.side_effect = RuntimeError("boom")
            openai_cls.return_value = client

            with pytest.raises(TranscriptionError):
                transcribe_openai(fake_audio, fake_metadata, cfg=cfg)
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_openai_backend_call.py -v
```

Expected: FAIL.

- [ ] **Step 3: Append `transcribe_openai()` to `src/yt2md/stages/transcribe_backends/openai.py`**

```python
from pathlib import Path

from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from yt2md.config import Config
from yt2md.errors import TranscriptionError
from yt2md.models import VideoMetadata
from yt2md.vocab_hint import extract_hints, format_for_openai


@retry(
    retry=retry_if_exception_type(
        (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
    ),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _call_openai_transcribe(
    client: OpenAI,
    audio_path: Path,
    model: str,
    prompt: str | None,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "file": audio_path.open("rb"),
        "response_format": "verbose_json",
        "timestamp_granularities": ["word", "segment"],
    }
    if prompt:
        kwargs["prompt"] = prompt
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
    except Exception as e:  # noqa: BLE001 -- mapping to typed exception
        msg = f"OpenAI transcribe failed: {e}"
        raise TranscriptionError(msg) from e

    raw = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    transcript = normalize_openai_response(raw, model_id=cfg.transcription_model)
    return transcript, raw
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_openai_backend_call.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/transcribe_backends/openai.py tests/unit/test_openai_backend_call.py
git commit -m "$(cat <<'EOF'
feat(transcribe-openai): SDK call with vocab hint and tenacity retries

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section E: transcribe_backends/local.py (faster-whisper)

---

### Task E.1: Local backend adapter + call

**Files:**
- Create: `tests/unit/test_local_backend.py`
- Create: `src/yt2md/stages/transcribe_backends/local.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_local_backend.py`:

```python
"""Tests for transcribe_local() — faster-whisper mocked."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from yt2md.config import Config
from yt2md.errors import ConfigError, TranscriptionError
from yt2md.stages.transcribe_backends.local import transcribe_local


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        google_api_key="g",  # type: ignore[arg-type]
        cache_dir=tmp_path,
        output_dir=tmp_path / "out",
        local_whisper_model="tiny",
    )


@pytest.fixture
def fake_audio(tmp_path: Path) -> Path:
    p = tmp_path / "a.opus"
    p.write_bytes(b"\x00" * 100)
    return p


@pytest.fixture
def fake_metadata(huberman_metadata):  # type: ignore[no-untyped-def]
    return huberman_metadata


def _fake_segment(start: float, end: float, text: str) -> SimpleNamespace:
    word = SimpleNamespace(word=text.strip(), start=start, end=end)
    return SimpleNamespace(
        start=start,
        end=end,
        text=text,
        words=[word],
    )


@pytest.fixture
def fake_segments() -> list[SimpleNamespace]:
    return [
        _fake_segment(0.0, 5.0, "Hello world."),
        _fake_segment(5.0, 10.0, "This is local whisper."),
    ]


@pytest.fixture
def fake_info() -> SimpleNamespace:
    return SimpleNamespace(language="en", duration=10.0)


class TestTranscribeLocal:
    def test_returns_transcript(
        self,
        cfg: Config,
        fake_audio: Path,
        fake_metadata,  # type: ignore[no-untyped-def]
        fake_segments: list[SimpleNamespace],
        fake_info: SimpleNamespace,
    ) -> None:
        with patch("yt2md.stages.transcribe_backends.local.WhisperModel") as model_cls:
            model = MagicMock()
            model.transcribe.return_value = (iter(fake_segments), fake_info)
            model_cls.return_value = model

            t, raw = transcribe_local(fake_audio, fake_metadata, cfg=cfg)

        assert t.backend == "local_whisper"
        assert t.model_id.startswith("faster-whisper")
        assert t.duration_s == 10.0
        assert t.speakers == []
        assert isinstance(raw, dict)

    def test_segments_normalized(
        self,
        cfg: Config,
        fake_audio: Path,
        fake_metadata,  # type: ignore[no-untyped-def]
        fake_segments: list[SimpleNamespace],
        fake_info: SimpleNamespace,
    ) -> None:
        with patch("yt2md.stages.transcribe_backends.local.WhisperModel") as model_cls:
            model = MagicMock()
            model.transcribe.return_value = (iter(fake_segments), fake_info)
            model_cls.return_value = model

            t, _ = transcribe_local(fake_audio, fake_metadata, cfg=cfg)

        assert len(t.segments) == 2
        assert t.segments[0].text == "Hello world."
        assert t.segments[0].start == 0.0
        assert t.segments[0].end == 5.0

    def test_passes_initial_prompt(
        self,
        cfg: Config,
        fake_audio: Path,
        fake_metadata,  # type: ignore[no-untyped-def]
        fake_segments: list[SimpleNamespace],
        fake_info: SimpleNamespace,
    ) -> None:
        with patch("yt2md.stages.transcribe_backends.local.WhisperModel") as model_cls:
            model = MagicMock()
            model.transcribe.return_value = (iter(fake_segments), fake_info)
            model_cls.return_value = model

            transcribe_local(fake_audio, fake_metadata, cfg=cfg)

            kwargs = model.transcribe.call_args.kwargs
            assert "initial_prompt" in kwargs
            assert "Huberman" in kwargs["initial_prompt"]

    def test_raises_config_error_if_faster_whisper_missing(
        self, cfg: Config, fake_audio: Path, fake_metadata  # type: ignore[no-untyped-def]
    ) -> None:
        with patch("yt2md.stages.transcribe_backends.local._import_faster_whisper") as imp:
            imp.side_effect = ImportError("no module")
            with pytest.raises(ConfigError, match="faster-whisper"):
                transcribe_local(fake_audio, fake_metadata, cfg=cfg)
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_local_backend.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write `src/yt2md/stages/transcribe_backends/local.py`**

```python
"""faster-whisper local transcription backend.

Optional dependency: import guarded so the module loads without the [local] extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from yt2md.config import Config
from yt2md.errors import ConfigError, TranscriptionError
from yt2md.models import Segment, Transcript, VideoMetadata, Word
from yt2md.vocab_hint import extract_hints, format_for_whisper


def _import_faster_whisper() -> Any:
    """Import faster_whisper.WhisperModel; isolated for mocking + clear error."""
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    return WhisperModel


# Re-exported for tests to patch.
try:
    WhisperModel = _import_faster_whisper()
except ImportError:
    WhisperModel = None  # type: ignore[assignment]


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
    except Exception as e:  # noqa: BLE001
        msg = f"Local whisper transcribe failed: {e}"
        raise TranscriptionError(msg) from e

    model_id = f"faster-whisper-{cfg.local_whisper_model}"
    transcript = _normalize_local_response(segments_list, info, model_id=model_id)
    raw = _serialize_local_response(segments_list, info)
    return transcript, raw


def _normalize_local_response(
    segments_raw: list[Any], info: Any, *, model_id: str
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


def _normalize_segment(seg: Any) -> Segment:
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


def _serialize_local_response(segments: list[Any], info: Any) -> dict[str, Any]:
    return {
        "language": getattr(info, "language", "en"),
        "duration": getattr(info, "duration", 0.0),
        "segments": [
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": str(s.text),
                "words": [
                    {"word": str(getattr(w, "word", "")), "start": float(w.start), "end": float(w.end)}
                    for w in (getattr(s, "words", None) or [])
                ],
            }
            for s in segments
        ],
    }
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_local_backend.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/transcribe_backends/local.py tests/unit/test_local_backend.py
git commit -m "$(cat <<'EOF'
feat(transcribe-local): faster-whisper backend with initial_prompt and import guard

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section F: chunk.py (silence-boundary chunking + stitching)

---

### Task F.1: Stitch helper — concatenate transcripts with offsets

**Files:**
- Create: `tests/unit/test_chunk_stitch.py`
- Create: `src/yt2md/stages/chunk.py`

The stitch logic is testable without ffmpeg. Build it first.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_chunk_stitch.py`:

```python
"""Tests for chunk.stitch() — combine per-chunk transcripts with offset timestamps."""

from yt2md.models import Segment, Transcript, Word
from yt2md.stages.chunk import stitch_transcripts


def _t(start: float, end: float, segments: list[Segment]) -> Transcript:
    return Transcript(
        language="en",
        duration_s=end - start,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=segments,
        speakers=[],
    )


def _seg(start: float, end: float, text: str) -> Segment:
    return Segment(
        start=start,
        end=end,
        text=text,
        speaker=None,
        words=[Word(text=text, start=start, end=end, speaker=None)],
    )


class TestStitch:
    def test_single_chunk_passthrough(self) -> None:
        t = _t(0.0, 10.0, [_seg(0.0, 10.0, "hi")])
        stitched = stitch_transcripts([t], offsets_s=[0.0])
        assert stitched == t.model_copy(update={"chunked": True})

    def test_two_chunks_offset_applied(self) -> None:
        c1 = _t(0.0, 30.0, [_seg(0.0, 10.0, "a"), _seg(10.0, 30.0, "b")])
        c2 = _t(0.0, 20.0, [_seg(0.0, 20.0, "c")])
        stitched = stitch_transcripts([c1, c2], offsets_s=[0.0, 30.0])

        starts = [s.start for s in stitched.segments]
        assert starts == [0.0, 10.0, 30.0]

        ends = [s.end for s in stitched.segments]
        assert ends == [10.0, 30.0, 50.0]

    def test_word_timestamps_also_offset(self) -> None:
        c1 = _t(0.0, 5.0, [_seg(0.0, 5.0, "a")])
        c2 = _t(0.0, 5.0, [_seg(0.0, 5.0, "b")])
        stitched = stitch_transcripts([c1, c2], offsets_s=[0.0, 5.0])
        c2_word = stitched.segments[1].words[0]
        assert c2_word.start == 5.0
        assert c2_word.end == 10.0

    def test_chunked_flag_true(self) -> None:
        t = _t(0.0, 1.0, [_seg(0.0, 1.0, "x")])
        stitched = stitch_transcripts([t], offsets_s=[0.0])
        assert stitched.chunked is True

    def test_total_duration_is_max_end(self) -> None:
        c1 = _t(0.0, 30.0, [_seg(0.0, 30.0, "a")])
        c2 = _t(0.0, 20.0, [_seg(0.0, 20.0, "b")])
        stitched = stitch_transcripts([c1, c2], offsets_s=[0.0, 30.0])
        assert stitched.duration_s == 50.0
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_chunk_stitch.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write `src/yt2md/stages/chunk.py`**

```python
"""Chunking for long audio: silence-boundary split + offset stitching.

Public surface:
  - needs_chunking(audio, backend, cfg) → bool
  - split_at_silence(audio, backend, cfg) → list[Chunk]
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
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_chunk_stitch.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/chunk.py tests/unit/test_chunk_stitch.py
git commit -m "$(cat <<'EOF'
feat(chunk): stitch_transcripts() applies per-chunk offsets to timestamps

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task F.2: needs_chunking() and split_at_silence() — ffmpeg silence detection mocked

**Files:**
- Create: `tests/unit/test_chunk_split.py`
- Modify: `src/yt2md/stages/chunk.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_chunk_split.py`:

```python
"""Tests for needs_chunking() and split_at_silence() — ffmpeg/ffprobe mocked."""

from pathlib import Path
from unittest.mock import patch

import pytest

from yt2md.config import Config
from yt2md.stages.chunk import needs_chunking, split_at_silence


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(google_api_key="g", cache_dir=tmp_path)  # type: ignore[arg-type]


class TestNeedsChunking:
    def test_small_file_no_chunking(self, cfg: Config, tmp_path: Path) -> None:
        audio = tmp_path / "small.opus"
        audio.write_bytes(b"\x00" * (5 * 1024 * 1024))  # 5MB
        with patch("yt2md.stages.chunk._ffprobe_duration", return_value=600.0):
            assert needs_chunking(audio, backend="openai_transcribe", cfg=cfg) is False

    def test_large_file_chunks(self, cfg: Config, tmp_path: Path) -> None:
        audio = tmp_path / "large.opus"
        audio.write_bytes(b"\x00" * (25 * 1024 * 1024))  # 25MB
        with patch("yt2md.stages.chunk._ffprobe_duration", return_value=600.0):
            assert needs_chunking(audio, backend="openai_transcribe", cfg=cfg) is True

    def test_long_duration_chunks_even_if_small_file(self, cfg: Config, tmp_path: Path) -> None:
        audio = tmp_path / "long.opus"
        audio.write_bytes(b"\x00" * 1000)  # tiny but long-duration
        with patch("yt2md.stages.chunk._ffprobe_duration", return_value=4 * 3600.0):
            # 4 hours exceeds typical model cap
            assert needs_chunking(audio, backend="openai_transcribe", cfg=cfg) is True


class TestSplitAtSilence:
    def test_returns_chunks_with_paths_and_offsets(
        self, cfg: Config, tmp_path: Path
    ) -> None:
        audio = tmp_path / "in.opus"
        audio.write_bytes(b"\x00" * 1000)

        # Mock the duration + silence detection + ffmpeg cut
        with patch("yt2md.stages.chunk._ffprobe_duration", return_value=3600.0), \
             patch("yt2md.stages.chunk._detect_silences", return_value=[1200.0, 2400.0]), \
             patch("yt2md.stages.chunk._cut_chunk") as cut:
            chunks = split_at_silence(audio, backend="openai_transcribe", cfg=cfg)

        assert len(chunks) == 3
        assert [c.start_offset_s for c in chunks] == [0.0, 1200.0, 2400.0]
        assert cut.call_count == 3
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_chunk_split.py -v
```

Expected: FAIL.

- [ ] **Step 3: Append to `src/yt2md/stages/chunk.py`**

```python
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from yt2md.config import Config

# Hard caps per backend. OpenAI: file size 25MB (with 20% safety margin → 20MB)
# AND ~25 minute (1500s) duration per request — to be confirmed against current docs.
# Local: no hard cap but memory scales linearly; chunk >1hr to keep RAM bounded.
SIZE_CAP_MB = {
    "openai_transcribe": 20,
    "local_whisper": 200,  # generous; mostly for memory-bound users
}
DURATION_CAP_S = {
    "openai_transcribe": 1500.0,  # 25 min
    "local_whisper": 3600.0,  # 1 hr
}

SILENCE_SEARCH_WINDOW_S = 30.0
SILENCE_MIN_S = 0.5
SILENCE_NOISE_DB = -30


@dataclass(frozen=True)
class Chunk:
    path: Path
    start_offset_s: float
    duration_s: float


def needs_chunking(audio: Path, *, backend: str, cfg: Config) -> bool:
    """Decide if chunking is required for this backend."""
    size_mb = audio.stat().st_size / (1024 * 1024)
    if size_mb > SIZE_CAP_MB.get(backend, 200):
        return True
    duration = _ffprobe_duration(audio)
    return duration > DURATION_CAP_S.get(backend, 3600.0)


def split_at_silence(audio: Path, *, backend: str, cfg: Config) -> list[Chunk]:
    """Split `audio` into ~80%-of-cap chunks at silence boundaries.

    Cut points: ideal-boundary ± SILENCE_SEARCH_WINDOW_S, longest silence wins.
    Falls back to ideal boundary if no silence detected in window (rare).
    """
    duration = _ffprobe_duration(audio)
    target_chunk_s = DURATION_CAP_S[backend] * 0.8
    num_chunks = max(1, int(duration / target_chunk_s) + (1 if duration % target_chunk_s else 0))
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
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(audio),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
    return float(result.stdout.strip())


def _detect_silences(audio: Path) -> list[float]:
    """Run ffmpeg silencedetect; parse silence_start timestamps from stderr."""
    cmd = [
        "ffmpeg", "-i", str(audio),
        "-af", f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_S}",
        "-f", "null", "-",
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
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-i", str(source),
        "-t", str(duration_s),
        "-c", "copy",
        str(destination),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)  # noqa: S603
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_chunk_split.py -v
uv run pytest tests/unit/test_chunk_stitch.py -v
```

Expected: all PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/chunk.py tests/unit/test_chunk_split.py
git commit -m "$(cat <<'EOF'
feat(chunk): needs_chunking + split_at_silence with ffmpeg silencedetect

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task F.3: transcribe() dispatcher integrates chunking

**Files:**
- Create: `tests/unit/test_transcribe_integration.py`
- Modify: `src/yt2md/stages/transcribe.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_transcribe_integration.py`:

```python
"""Tests for transcribe() — backend dispatch with chunking integration."""

from pathlib import Path
from unittest.mock import patch

import pytest

from yt2md.config import Config
from yt2md.models import Segment, Transcript, Word
from yt2md.stages.transcribe import transcribe


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        google_api_key="g",  # type: ignore[arg-type]
        openai_api_key="okey",  # type: ignore[arg-type]
        cache_dir=tmp_path,
        output_dir=tmp_path / "out",
    )


@pytest.fixture
def fake_audio(tmp_path: Path) -> Path:
    p = tmp_path / "audio.opus"
    p.write_bytes(b"\x00" * 1000)
    return p


def _tx(duration: float, text: str) -> Transcript:
    return Transcript(
        language="en",
        duration_s=duration,
        backend="openai_transcribe",
        model_id="gpt-4o-transcribe",
        chunked=False,
        segments=[
            Segment(
                start=0.0, end=duration, text=text, speaker=None,
                words=[Word(text=text, start=0.0, end=duration, speaker=None)],
            ),
        ],
        speakers=[],
    )


class TestNoChunking:
    def test_single_pass(self, cfg: Config, fake_audio: Path, huberman_metadata) -> None:  # type: ignore[no-untyped-def]
        t_ret = _tx(60.0, "no chunking happened")
        with patch("yt2md.stages.transcribe.needs_chunking", return_value=False), \
             patch("yt2md.stages.transcribe.transcribe_openai", return_value=(t_ret, {})):
            result, raw = transcribe(fake_audio, huberman_metadata, cfg=cfg)
        assert result.chunked is False
        assert result.segments[0].text == "no chunking happened"


class TestChunking:
    def test_multi_chunk_stitched(self, cfg: Config, fake_audio: Path, huberman_metadata) -> None:  # type: ignore[no-untyped-def]
        from yt2md.stages.chunk import Chunk

        chunk1 = Chunk(path=fake_audio, start_offset_s=0.0, duration_s=30.0)
        chunk2 = Chunk(path=fake_audio, start_offset_s=30.0, duration_s=30.0)

        with patch("yt2md.stages.transcribe.needs_chunking", return_value=True), \
             patch("yt2md.stages.transcribe.split_at_silence", return_value=[chunk1, chunk2]), \
             patch("yt2md.stages.transcribe.transcribe_openai",
                   side_effect=[(_tx(30.0, "first"), {}), (_tx(30.0, "second"), {})]):
            result, raw = transcribe(fake_audio, huberman_metadata, cfg=cfg)

        assert result.chunked is True
        assert len(result.segments) == 2
        # Second chunk's segment was offset by 30s
        assert result.segments[1].start == 30.0
        assert result.segments[1].end == 60.0
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_transcribe_integration.py -v
```

Expected: FAIL — `transcribe()` not yet exported.

- [ ] **Step 3: Append to `src/yt2md/stages/transcribe.py`**

```python
from pathlib import Path
from typing import Any

from yt2md.models import Transcript, VideoMetadata
from yt2md.stages.chunk import Chunk, needs_chunking, split_at_silence, stitch_transcripts
from yt2md.stages.transcribe_backends.local import transcribe_local
from yt2md.stages.transcribe_backends.openai import transcribe_openai


def transcribe(
    audio: Path,
    metadata: VideoMetadata,
    *,
    cfg: Config,
) -> tuple[Transcript, list[dict[str, Any]]]:
    """Transcribe `audio`. Dispatches to backend, chunks if needed.

    Returns (stitched_transcript, list_of_raw_responses).
    """
    backend = resolve_backend(cfg)
    backend_fn = _backend_function(backend)

    if not needs_chunking(audio, backend=backend, cfg=cfg):
        transcript, raw = backend_fn(audio, metadata, cfg=cfg)
        return transcript, [raw]

    chunks = split_at_silence(audio, backend=backend, cfg=cfg)
    chunk_transcripts: list[Transcript] = []
    chunk_raws: list[dict[str, Any]] = []
    for chunk in chunks:
        t, raw = backend_fn(chunk.path, metadata, cfg=cfg)
        chunk_transcripts.append(t)
        chunk_raws.append(raw)

    stitched = stitch_transcripts(
        chunk_transcripts,
        offsets_s=[c.start_offset_s for c in chunks],
    )
    return stitched, chunk_raws


def _backend_function(backend: ResolvedBackend):  # type: ignore[no-untyped-def]
    if backend == "openai_transcribe":
        return transcribe_openai
    if backend == "local_whisper":
        return transcribe_local
    msg = f"Unknown resolved backend: {backend}"
    raise ConfigError(msg)
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_transcribe_integration.py -v
uv run pytest tests/unit/test_transcribe_dispatch.py -v
```

Expected: all PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/transcribe.py tests/unit/test_transcribe_integration.py
git commit -m "$(cat <<'EOF'
feat(transcribe): dispatcher with chunking + stitch integration

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section G: structure.py (Gemini structured output)

---

### Task G.1: Prompt builder — transcript serialization with [mm:ss] markers

**Files:**
- Create: `tests/unit/test_structure_prompt.py`
- Create: `src/yt2md/stages/structure.py`
- Create: `src/yt2md/prompts/structure.md`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_structure_prompt.py`:

```python
"""Tests for build_structure_prompt() — transcript with inline timestamps."""

from yt2md.models import Segment, Transcript, Word
from yt2md.stages.structure import build_structure_prompt


def _t(*, segments: list[Segment]) -> Transcript:
    duration = max((s.end for s in segments), default=0.0)
    return Transcript(
        language="en",
        duration_s=duration,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=segments,
        speakers=["SPEAKER_00"],
    )


def _seg(start: float, end: float, text: str, speaker: str | None = "SPEAKER_00") -> Segment:
    return Segment(
        start=start, end=end, text=text, speaker=speaker,
        words=[Word(text=text, start=start, end=end, speaker=speaker)],
    )


class TestPromptStructure:
    def test_contains_metadata_section(self, huberman_metadata) -> None:  # type: ignore[no-untyped-def]
        t = _t(segments=[_seg(0.0, 5.0, "hello")])
        prompt = build_structure_prompt(t, huberman_metadata)
        assert "title" in prompt.lower()
        assert "Huberman Lab" in prompt

    def test_contains_transcript_section(self, huberman_metadata) -> None:  # type: ignore[no-untyped-def]
        t = _t(segments=[_seg(252.0, 260.0, "Dopamine signals anticipation.")])
        prompt = build_structure_prompt(t, huberman_metadata)
        # Inline [mm:ss] marker for 252s = 04:12
        assert "[04:12]" in prompt
        assert "Dopamine signals anticipation." in prompt

    def test_chunked_flag_propagated(self, huberman_metadata) -> None:  # type: ignore[no-untyped-def]
        t = _t(segments=[_seg(0.0, 5.0, "x")])
        t_chunked = t.model_copy(update={"chunked": True})
        prompt = build_structure_prompt(t_chunked, huberman_metadata)
        # The prompt should warn the model about speaker labels being unreliable across chunks
        assert "chunk" in prompt.lower()
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_structure_prompt.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write `src/yt2md/prompts/structure.md`**

```markdown
---
version: 1
---

You are extracting a structured knowledge-graph node from a YouTube video transcript.

# Inputs

You are given:
- Video metadata (title, channel, published date, description, chapters)
- A cleaned transcript of the spoken content, with `[mm:ss]` timestamp markers at the start of each segment and speaker labels (SPEAKER_00, SPEAKER_01, ...) where diarization succeeded.

# Task

Produce a JSON document matching the supplied schema. The document should be dense, detailed, and self-contained. Each section is described below.

## frontmatter

- `title`: the video's title verbatim.
- `channel`: the channel name verbatim.
- `url`, `video_id`, `published`, `duration_seconds`, `captured_at`, `schema_version`: copy from the input metadata; we'll override these if needed.
- `genre`: classify the video as one of: podcast, lecture, tutorial, talk, interview, other.
- `speakers`: human names of the speakers (mapped from SPEAKER_NN). For solo content, the host alone.
- `topics`: 3-7 high-level topic tags (lowercase, hyphenated multi-word: "habit-formation").
- `people_mentioned`: names of people referenced in the content but not present as speakers.
- `works_mentioned`: books, papers, products, or other named works cited.

## speaker_name_map

A dictionary mapping `SPEAKER_NN` → human name (e.g., `{"SPEAKER_00": "Andrew Huberman"}`). Infer names from the transcript content (introductions, self-references, channel name). Leave empty if the transcript is undiarized.

## tldr

A 3-5 sentence dense summary. Self-contained — name the speaker(s) and topic, so a RAG chunk pulled in isolation remains grounded.

## takeaways

3-8 short, dense bullet points capturing the most important claims. Each has a `text` and `timestamp_s` (the start time of the segment containing the claim).

## concepts

Named concepts or definitions introduced. Each has `name`, `definition` (1-2 sentences), `timestamp_s`.

## references

People, books, papers, tools, or videos referenced. Each has `kind` (one of: book, paper, person, tool, video, other), `name`, `context` (1-2 sentence summary of what was said about it), `timestamp_s`.

## quotes

Verbatim quotes worth surfacing. Each has `text`, `speaker` (mapped name), `timestamp_s`. Quote sparingly — only when the exact wording matters.

## sections

Detailed notes broken into logical sections. Each has `heading`, `body` (2-4 paragraphs of dense prose, self-contained: re-name the speaker and topic), `timestamp_s` (section start).

## open_questions

Questions raised but not answered in the video, flagged for future exploration. Empty list if none.

# Rules

- Timestamps: copy the `[mm:ss]` marker shown at the start of each segment into `timestamp_s` as float seconds (e.g., `[04:12]` → `252.0`).
- Be faithful to the transcript. Do not invent claims, quotes, or references not present in the content.
- For chunked transcripts (`Transcript.chunked = true`), speaker labels may be inconsistent across the document. Use named identity (from frontmatter `speakers`) when uncertain.

---

# Metadata

{{ metadata_block }}

# Transcript

{{ transcript_block }}
```

- [ ] **Step 4: Write `src/yt2md/stages/structure.py`**

```python
"""Structure stage: build Gemini prompt, call Gemini, validate, retry once."""

from __future__ import annotations

from importlib import resources

from yt2md.models import Transcript, VideoMetadata

PROMPT_VERSION = 1


def build_structure_prompt(transcript: Transcript, metadata: VideoMetadata) -> str:
    """Render the structuring prompt with metadata and inline-timestamped transcript."""
    template = (
        resources.files("yt2md") / "prompts" / "structure.md"
    ).read_text(encoding="utf-8")
    metadata_block = _format_metadata(metadata)
    transcript_block = _format_transcript(transcript)
    rendered = template.replace("{{ metadata_block }}", metadata_block)
    rendered = rendered.replace("{{ transcript_block }}", transcript_block)
    if transcript.chunked:
        rendered += (
            "\n\nNote: This transcript was produced from multiple chunks. "
            "Speaker labels may be inconsistent across chunks; refer to speakers "
            "by their named identity (from metadata) when in doubt.\n"
        )
    return rendered


def _format_metadata(meta: VideoMetadata) -> str:
    chapters = "\n".join(
        f"  - {c.title} ({_mmss(c.start_s)}-{_mmss(c.end_s)})" for c in meta.chapters
    ) or "  (none)"
    return (
        f"title: {meta.title}\n"
        f"channel: {meta.channel}\n"
        f"published: {meta.published_date}\n"
        f"duration_seconds: {int(meta.duration_s)}\n"
        f"url: {meta.url}\n"
        f"video_id: {meta.video_id}\n"
        f"description: |\n  {meta.description[:1000]}\n"
        f"chapters:\n{chapters}\n"
    )


def _format_transcript(t: Transcript) -> str:
    lines: list[str] = []
    for seg in t.segments:
        prefix = f"[{_mmss(seg.start)}]"
        if seg.speaker:
            lines.append(f"{prefix} {seg.speaker}: {seg.text}")
        else:
            lines.append(f"{prefix} {seg.text}")
    return "\n".join(lines)


def _mmss(seconds_value: float) -> str:
    total = int(seconds_value)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
```

- [ ] **Step 5: Update pyproject to ship prompts/**

In `pyproject.toml` `[tool.hatch.build.targets.wheel.force-include]`:

```toml
"src/yt2md/templates/document.md.j2" = "yt2md/templates/document.md.j2"
"src/yt2md/prompts/structure.md" = "yt2md/prompts/structure.md"
```

Also create `src/yt2md/prompts/__init__.py` empty marker:

```bash
touch src/yt2md/prompts/__init__.py
```

- [ ] **Step 6: Run — confirm passes**

```bash
uv run pytest tests/unit/test_structure_prompt.py -v
```

Expected: 3 PASS.

- [ ] **Step 7: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/structure.py src/yt2md/prompts/ tests/unit/test_structure_prompt.py pyproject.toml
git commit -m "$(cat <<'EOF'
feat(structure): prompt builder with inline mm:ss timestamps and PROMPT_VERSION=1

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task G.2: Semantic validation — timestamps in range, takeaways non-empty

**Files:**
- Create: `tests/unit/test_structure_validation.py`
- Modify: `src/yt2md/stages/structure.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_structure_validation.py`:

```python
"""Tests for validate_structured_doc — semantic checks beyond Pydantic shape."""

from datetime import date

import pytest

from yt2md.errors import InvalidStructuredOutputError
from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    Concept,
    DetailedSection,
    Frontmatter,
    Quote,
    Reference,
    StructuredDoc,
    Takeaway,
    Transcript,
    VideoMetadata,
)
from yt2md.stages.structure import validate_structured_doc


def _doc(
    *,
    title: str = "T",
    video_id: str = "vid",
    takeaways: list[Takeaway] | None = None,
    tldr: str = "Non-empty.",
    quotes: list[Quote] | None = None,
    speaker_name_map: dict[str, str] | None = None,
) -> StructuredDoc:
    return StructuredDoc(
        frontmatter=Frontmatter(
            title=title,
            channel="C",
            url="u",
            video_id=video_id,
            published=date(2025, 1, 1),
            duration_seconds=10,
            captured_at=date(2026, 5, 23),
            schema_version=CURRENT_SCHEMA_VERSION,
            genre="podcast",
            speakers=["A"],
            topics=[],
            people_mentioned=[],
            works_mentioned=[],
        ),
        tldr=tldr,
        takeaways=takeaways or [Takeaway(text="a", timestamp_s=0.0)] * 3,
        concepts=[],
        references=[],
        quotes=quotes or [],
        sections=[],
        open_questions=[],
        speaker_name_map=speaker_name_map or {},
    )


def _meta(title: str = "T", video_id: str = "vid", duration_s: float = 10.0) -> VideoMetadata:
    return VideoMetadata(
        video_id=video_id,
        url="u",
        title=title,
        channel="C",
        channel_id="UC",
        published_date=date(2025, 1, 1),
        duration_s=duration_s,
        description="",
        chapters=[],
        tags=[],
        language=None,
    )


def _transcript(duration_s: float = 10.0) -> Transcript:
    return Transcript(
        language="en",
        duration_s=duration_s,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=[],
        speakers=[],
    )


class TestValidationSuccess:
    def test_minimal_valid_doc(self) -> None:
        validate_structured_doc(_doc(), transcript=_transcript(), metadata=_meta())


class TestRequiredFields:
    def test_takeaways_must_be_3_or_more(self) -> None:
        d = _doc(takeaways=[Takeaway(text="x", timestamp_s=0.0)])
        with pytest.raises(InvalidStructuredOutputError, match="takeaways"):
            validate_structured_doc(d, transcript=_transcript(), metadata=_meta())

    def test_tldr_nonempty(self) -> None:
        d = _doc(tldr="   ")
        with pytest.raises(InvalidStructuredOutputError, match="tldr"):
            validate_structured_doc(d, transcript=_transcript(), metadata=_meta())


class TestFrontmatterConsistency:
    def test_title_matches_metadata(self) -> None:
        d = _doc(title="Mismatch")
        with pytest.raises(InvalidStructuredOutputError, match="title"):
            validate_structured_doc(d, transcript=_transcript(), metadata=_meta(title="Real"))

    def test_video_id_matches_metadata(self) -> None:
        d = _doc(video_id="X")
        with pytest.raises(InvalidStructuredOutputError, match="video_id"):
            validate_structured_doc(d, transcript=_transcript(), metadata=_meta(video_id="Y"))


class TestTimestampRange:
    def test_takeaway_timestamp_in_range(self) -> None:
        d = _doc(takeaways=[
            Takeaway(text="x", timestamp_s=0.0),
            Takeaway(text="y", timestamp_s=5.0),
            Takeaway(text="z", timestamp_s=100.0),  # > duration_s
        ])
        with pytest.raises(InvalidStructuredOutputError, match="timestamp"):
            validate_structured_doc(d, transcript=_transcript(duration_s=10.0), metadata=_meta())

    def test_quote_timestamp_in_range(self) -> None:
        d = _doc(quotes=[Quote(text="q", speaker="A", timestamp_s=999.0)])
        with pytest.raises(InvalidStructuredOutputError, match="timestamp"):
            validate_structured_doc(d, transcript=_transcript(duration_s=10.0), metadata=_meta())
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_structure_validation.py -v
```

Expected: FAIL.

- [ ] **Step 3: Append to `src/yt2md/stages/structure.py`**

```python
from yt2md.errors import InvalidStructuredOutputError
from yt2md.models import StructuredDoc

MIN_TAKEAWAYS = 3


def validate_structured_doc(
    doc: StructuredDoc,
    *,
    transcript: Transcript,
    metadata: VideoMetadata,
) -> None:
    """Semantic validation beyond Pydantic's shape checks. Raises on failure."""
    if len(doc.takeaways) < MIN_TAKEAWAYS:
        msg = f"takeaways: need at least {MIN_TAKEAWAYS}, got {len(doc.takeaways)}"
        raise InvalidStructuredOutputError(msg)

    if not doc.tldr.strip():
        msg = "tldr is empty"
        raise InvalidStructuredOutputError(msg)

    if doc.frontmatter.title != metadata.title:
        msg = f"frontmatter.title {doc.frontmatter.title!r} != metadata.title {metadata.title!r}"
        raise InvalidStructuredOutputError(msg)

    if doc.frontmatter.video_id != metadata.video_id:
        msg = (
            f"frontmatter.video_id {doc.frontmatter.video_id!r} != "
            f"metadata.video_id {metadata.video_id!r}"
        )
        raise InvalidStructuredOutputError(msg)

    duration = transcript.duration_s
    _check_timestamps([t.timestamp_s for t in doc.takeaways], duration, "takeaways")
    _check_timestamps([c.timestamp_s for c in doc.concepts], duration, "concepts")
    _check_timestamps([r.timestamp_s for r in doc.references], duration, "references")
    _check_timestamps([q.timestamp_s for q in doc.quotes], duration, "quotes")
    _check_timestamps([s.timestamp_s for s in doc.sections], duration, "sections")


def _check_timestamps(stamps: list[float], duration_s: float, field_name: str) -> None:
    for ts in stamps:
        if ts < 0.0 or ts > duration_s:
            msg = f"{field_name}: timestamp {ts} out of range [0, {duration_s}]"
            raise InvalidStructuredOutputError(msg)
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_structure_validation.py -v
```

Expected: 7 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/structure.py tests/unit/test_structure_validation.py
git commit -m "$(cat <<'EOF'
feat(structure): validate_structured_doc — semantic checks beyond Pydantic shape

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task G.3: structure() — Gemini call with retry-on-validation-failure

**Files:**
- Create: `tests/unit/test_structure_call.py`
- Modify: `src/yt2md/stages/structure.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_structure_call.py`:

```python
"""Tests for structure() — Gemini call mocked + validation retry semantics."""

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from yt2md.config import Config
from yt2md.errors import InvalidStructuredOutputError
from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    StructuredDoc,
    Takeaway,
    Transcript,
    VideoMetadata,
)
from yt2md.stages.structure import structure


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        google_api_key="g",  # type: ignore[arg-type]
        cache_dir=tmp_path,
    )


def _meta() -> VideoMetadata:
    return VideoMetadata(
        video_id="vid",
        url="u",
        title="T",
        channel="C",
        channel_id="UC",
        published_date=date(2025, 1, 1),
        duration_s=60.0,
        description="",
        chapters=[],
        tags=[],
        language=None,
    )


def _transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_s=60.0,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=[],
        speakers=[],
    )


def _valid_response_json() -> str:
    return json.dumps({
        "frontmatter": {
            "title": "T",
            "channel": "C",
            "url": "u",
            "video_id": "vid",
            "published": "2025-01-01",
            "duration_seconds": 60,
            "captured_at": "2026-05-23",
            "schema_version": CURRENT_SCHEMA_VERSION,
            "genre": "podcast",
            "speakers": ["A"],
            "topics": [],
            "people_mentioned": [],
            "works_mentioned": [],
        },
        "tldr": "Hello.",
        "takeaways": [
            {"text": "t1", "timestamp_s": 0.0},
            {"text": "t2", "timestamp_s": 1.0},
            {"text": "t3", "timestamp_s": 2.0},
        ],
        "concepts": [],
        "references": [],
        "quotes": [],
        "sections": [],
        "open_questions": [],
        "speaker_name_map": {},
    })


def _invalid_response_json() -> str:
    # Only 1 takeaway → validation fails (need ≥3)
    data = json.loads(_valid_response_json())
    data["takeaways"] = [{"text": "only-one", "timestamp_s": 0.0}]
    return json.dumps(data)


def _fake_gemini_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


class TestStructureHappy:
    def test_returns_structured_doc(self, cfg: Config) -> None:
        with patch("yt2md.stages.structure._call_gemini") as call:
            call.return_value = _fake_gemini_response(_valid_response_json())
            doc = structure(_transcript(), _meta(), cfg=cfg)
        assert isinstance(doc, StructuredDoc)
        assert len(doc.takeaways) == 3


class TestStructureRetryOnValidation:
    def test_retries_once_on_invalid_then_succeeds(self, cfg: Config) -> None:
        responses = [
            _fake_gemini_response(_invalid_response_json()),
            _fake_gemini_response(_valid_response_json()),
        ]
        with patch("yt2md.stages.structure._call_gemini", side_effect=responses) as call:
            doc = structure(_transcript(), _meta(), cfg=cfg)
        assert call.call_count == 2
        assert isinstance(doc, StructuredDoc)

    def test_raises_after_second_failure(self, cfg: Config) -> None:
        with patch("yt2md.stages.structure._call_gemini") as call:
            call.return_value = _fake_gemini_response(_invalid_response_json())
            with pytest.raises(InvalidStructuredOutputError):
                structure(_transcript(), _meta(), cfg=cfg)
            assert call.call_count == 2
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_structure_call.py -v
```

Expected: FAIL.

- [ ] **Step 3: Append to `src/yt2md/stages/structure.py`**

```python
import json
from typing import Any

from google import genai  # type: ignore[import-not-found]
from google.genai import types as genai_types  # type: ignore[import-not-found]
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from yt2md.config import Config

GEMINI_TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 20000


def structure(
    transcript: Transcript,
    metadata: VideoMetadata,
    *,
    cfg: Config,
) -> StructuredDoc:
    """Call Gemini for analytical extraction. Retry once on validation failure.

    Raises InvalidStructuredOutputError if the second attempt also fails.
    """
    prompt = build_structure_prompt(transcript, metadata)
    last_error: Exception | None = None

    for attempt in (1, 2):
        try:
            response = _call_gemini(prompt, cfg=cfg)
            raw = json.loads(response.text)
            doc = StructuredDoc.model_validate(raw)
            validate_structured_doc(doc, transcript=transcript, metadata=metadata)
        except (ValidationError, InvalidStructuredOutputError, json.JSONDecodeError) as e:
            last_error = e
            if attempt == 2:
                msg = f"Gemini output failed validation after retry: {e}"
                raise InvalidStructuredOutputError(msg) from e
            # Append validation feedback to the prompt and try again.
            prompt = prompt + f"\n\n# Previous attempt failed validation\n\n{e!s}\n\nPlease retry, fixing the issue above."
            continue
        else:
            return doc

    msg = f"Unreachable: last_error={last_error}"
    raise InvalidStructuredOutputError(msg)


def _call_gemini(prompt: str, *, cfg: Config) -> Any:
    """Single Gemini API call with tenacity retries on transient SDK errors."""
    return _call_gemini_inner(prompt, cfg)


@retry(
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _call_gemini_inner(prompt: str, cfg: Config) -> Any:
    client = genai.Client(api_key=cfg.google_api_key.get_secret_value())
    response = client.models.generate_content(
        model=cfg.structuring_model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=StructuredDoc.model_json_schema(),
            temperature=GEMINI_TEMPERATURE,
            seed=hash(prompt) % (2**31),
            max_output_tokens=MAX_OUTPUT_TOKENS,
            safety_settings=[
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE",
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE",
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE",
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE",
                ),
            ],
        ),
    )
    return response
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_structure_call.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/structure.py tests/unit/test_structure_call.py
git commit -m "$(cat <<'EOF'
feat(structure): Gemini call with response_schema + 1 retry on validation failure

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section H: Live tests for transcribe + structure (optional, gated)

---

### Task H.1: Live OpenAI transcribe test

**Files:**
- Create: `tests/live/test_live_transcribe_openai.py`
- Create: `tests/fixtures/audio/short_speech_30s.opus` (provide separately — LibriSpeech sample or hand-recorded; not generated by this plan)

- [ ] **Step 1: Provide the short audio fixture**

You need a 30-second speech audio file at `tests/fixtures/audio/short_speech_30s.opus`. Acquire one of:
- A LibriSpeech-derived clip (e.g., `dev-clean` subset, attributed)
- A self-recorded 30s sample
- Convert a public-domain TED-Ed clip

If none available right now, mark this task incomplete and circle back. Phase 4's end-to-end test depends on this fixture.

- [ ] **Step 2: Write the live test**

`tests/live/test_live_transcribe_openai.py`:

```python
"""Live test for OpenAI transcribe backend. Skipped without -m live + OPENAI_API_KEY."""

import os
from datetime import date
from pathlib import Path

import pytest

from yt2md.config import Config
from yt2md.models import VideoMetadata
from yt2md.stages.transcribe_backends.openai import transcribe_openai


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_live_transcribe_30s(fixtures_dir: Path, tmp_path: Path) -> None:
    audio = fixtures_dir / "audio" / "short_speech_30s.opus"
    if not audio.exists():
        pytest.skip(f"Fixture audio missing: {audio}")

    cfg = Config(
        google_api_key="g",  # type: ignore[arg-type]
        openai_api_key=os.environ["OPENAI_API_KEY"],  # type: ignore[arg-type]
        cache_dir=tmp_path,
        output_dir=tmp_path / "out",
    )
    meta = VideoMetadata(
        video_id="x",
        url="https://www.youtube.com/watch?v=x",
        title="Test",
        channel="Test",
        channel_id="UC",
        published_date=date(2025, 1, 1),
        duration_s=30.0,
        description="",
        chapters=[],
        tags=[],
        language="en",
    )
    transcript, raw = transcribe_openai(audio, meta, cfg=cfg)
    assert transcript.duration_s > 0
    assert len(transcript.segments) >= 1
    assert any(w.text for s in transcript.segments for w in s.words)
```

- [ ] **Step 3: Lint + commit (test will skip without setup)**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add tests/live/test_live_transcribe_openai.py
git commit -m "$(cat <<'EOF'
test(transcribe-openai): live test against 30s fixture (gated on -m live)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task H.2: Live Gemini structure test

**Files:**
- Create: `tests/live/test_live_structure_gemini.py`

- [ ] **Step 1: Write the live test**

`tests/live/test_live_structure_gemini.py`:

```python
"""Live test for Gemini structure stage. Skipped without -m live + GOOGLE_API_KEY."""

import json
import os
from pathlib import Path

import pytest

from yt2md.config import Config
from yt2md.models import Transcript, VideoMetadata
from yt2md.stages.structure import structure


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
def test_live_structure_small_transcript(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    transcript = Transcript.model_validate(
        json.loads((fixtures_dir / "transcripts" / "short_solo.json").read_text(encoding="utf-8"))
    )
    metadata = VideoMetadata.model_validate(
        json.loads((fixtures_dir / "metadata" / "huberman_sample.json").read_text(encoding="utf-8"))
    )
    # Resync to transcript's actual duration
    metadata = metadata.model_copy(update={"duration_s": transcript.duration_s})

    cfg = Config(
        google_api_key=os.environ["GOOGLE_API_KEY"],  # type: ignore[arg-type]
        cache_dir=tmp_path,
    )
    doc = structure(transcript, metadata, cfg=cfg)
    assert len(doc.takeaways) >= 3
    assert doc.tldr.strip()
    assert doc.frontmatter.title == metadata.title
```

- [ ] **Step 2: Lint + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add tests/live/test_live_structure_gemini.py
git commit -m "$(cat <<'EOF'
test(structure): live test against Gemini with fixture transcript

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 final check

### Task I.1: Run full Phase 3 suite + coverage

- [ ] **Step 1: Run all tests with coverage**

```bash
uv run pytest tests/unit tests/integration --cov=src/yt2md --cov-report=term-missing --cov-fail-under=85
```

Expected: all PASS; coverage ≥85%. SDK call paths (the actual `client.audio.transcriptions.create` invocations) are excluded via `# pragma: no cover` since they're tested live.

- [ ] **Step 2: Optional — run live tests if you have API keys**

```bash
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
uv run pytest tests/live -m live -v
```

Expected: 3 PASS (download + transcribe + structure live tests). Cost: <$0.05.

- [ ] **Step 3: Pre-commit all files**

```bash
uv run pre-commit run --all-files
```

Expected: all hooks pass.

- [ ] **Step 4: Mark Phase 3 complete in index**

```markdown
- [x] Phase 3 — API-bound stages
```

```bash
git add docs/superpowers/plans/2026-05-23-yt2llm-index.md
git commit -m "$(cat <<'EOF'
docs(plan): mark Phase 3 complete

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## What Phase 3 produced

- `src/yt2md/stages/download.py` — yt-dlp wrapper with metadata adapter, error mapping, cookie support
- `src/yt2md/stages/compress.py` — ffmpeg subprocess for canonical Opus 32kbps mono
- `src/yt2md/stages/transcribe.py` — backend dispatcher + chunking integration
- `src/yt2md/stages/transcribe_backends/openai.py` — gpt-4o-transcribe call + adapter + retries
- `src/yt2md/stages/transcribe_backends/local.py` — faster-whisper call + adapter + import guard
- `src/yt2md/stages/chunk.py` — silence-boundary chunker + stitcher
- `src/yt2md/stages/structure.py` — Gemini call with response_schema, validation, 1 retry
- `src/yt2md/prompts/structure.md` — versioned structuring prompt
- Live tests for each external API (skipped without keys)

**Still missing for MVP:** the pipeline orchestrator, CLI, idempotency logic, `regen` subcommand, observability glue (rich + structlog + runs.log), end-to-end integration. Those are Phase 4.

---

## Next: Phase 4

Open `docs/superpowers/plans/2026-05-23-yt2llm-phase-4-orchestrator-cli.md`.
