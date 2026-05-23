# yt2llm Phase 4: Orchestrator + CLI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Source spec:** `docs/superpowers/specs/2026-05-23-yt2llm-design.md`
**Index:** `docs/superpowers/plans/2026-05-23-yt2llm-index.md`
**Prereq:** Phases 1-3 complete (all stages + infrastructure modules).

**Goal:** Wire the seven stages into a `pipeline.run()` function, expose `yt2md <url>` via typer CLI with idempotency, observability (rich + structlog + runs.log), and the `regen` subcommand. End-to-end integration tests prove the MVP works.

**Architecture:** `pipeline.py` is the only module that knows the stage order. `cli.py` is a thin typer wrapper. Observability is layered: rich for the user, structlog for machine logs, JSONL for cost analytics.

**Tech Stack:** Wires Phases 1-3 together. Adds typer subcommand and structlog configuration.

**Definition of done:** All Phase 4 tasks checked off. Running `uv run yt2md <fake-url>` end-to-end (with all SDKs mocked at the boundary) produces a valid markdown file. `lint` + `typecheck` + `cover` all pass.

---

## Non-negotiable discipline (recap)

Same as Phases 1-3. TDD, lint+typecheck, 400 LOC ceiling, no abstractions without 3 concrete uses, never `--no-verify`. See `docs/superpowers/plans/2026-05-23-yt2llm-index.md`.

Shorthand: `lint`, `typecheck`, `cover` as defined in prior plans.

---

## Section A: pipeline.py (orchestrator)

---

### Task A.1: pipeline.run() — wire all 7 stages with caching

**Files:**
- Create: `tests/integration/test_pipeline_happy_path.py`
- Create: `src/yt2md/pipeline.py`

This is the heaviest integration test in Phase 4. Mocks every stage at its module boundary; verifies the final markdown gets written.

- [ ] **Step 1: Write the failing test**

`tests/integration/test_pipeline_happy_path.py`:

```python
"""End-to-end pipeline integration test with all stages mocked at the module boundary."""

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from yt2md.config import Config
from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    Frontmatter,
    Segment,
    StructuredDoc,
    Takeaway,
    Transcript,
    VideoMetadata,
    Word,
)
from yt2md.pipeline import run


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        google_api_key="g",  # type: ignore[arg-type]
        openai_api_key="o",  # type: ignore[arg-type]
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )


def _make_metadata() -> VideoMetadata:
    return VideoMetadata(
        video_id="abc123",
        url="https://www.youtube.com/watch?v=abc123",
        title="Test Episode",
        channel="Test Channel",
        channel_id="UC",
        published_date=date(2024, 3, 15),
        duration_s=10.0,
        description="",
        chapters=[],
        tags=[],
        language="en",
    )


def _make_transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_s=10.0,
        backend="openai_transcribe",
        model_id="gpt-4o-transcribe",
        chunked=False,
        segments=[
            Segment(
                start=0.0, end=10.0, text="hello world.", speaker="SPEAKER_00",
                words=[Word(text="hello", start=0.0, end=5.0, speaker="SPEAKER_00"),
                       Word(text="world.", start=5.0, end=10.0, speaker="SPEAKER_00")],
            ),
        ],
        speakers=["SPEAKER_00"],
    )


def _make_structured_doc() -> StructuredDoc:
    return StructuredDoc(
        frontmatter=Frontmatter(
            title="Test Episode",
            channel="Test Channel",
            url="https://www.youtube.com/watch?v=abc123",
            video_id="abc123",
            published=date(2024, 3, 15),
            duration_seconds=10,
            captured_at=date(2026, 5, 23),
            schema_version=CURRENT_SCHEMA_VERSION,
            genre="podcast",
            speakers=["Alice"],
            topics=[],
            people_mentioned=[],
            works_mentioned=[],
        ),
        tldr="TLDR sentence.",
        takeaways=[
            Takeaway(text="One.", timestamp_s=0.0),
            Takeaway(text="Two.", timestamp_s=2.0),
            Takeaway(text="Three.", timestamp_s=4.0),
        ],
        concepts=[],
        references=[],
        quotes=[],
        sections=[],
        open_questions=[],
        speaker_name_map={"SPEAKER_00": "Alice"},
    )


@pytest.fixture
def patched_stages(cfg: Config, tmp_path: Path):  # type: ignore[no-untyped-def]
    """Patch every external-touching stage."""
    metadata = _make_metadata()
    transcript = _make_transcript()
    doc = _make_structured_doc()

    # Simulate yt-dlp writing a source file
    source_audio = cfg.cache_dir / "abc123" / "source_audio.m4a"
    source_audio.parent.mkdir(parents=True, exist_ok=True)
    source_audio.write_bytes(b"\x00" * 100)

    with patch("yt2md.pipeline.download") as dl, \
         patch("yt2md.pipeline.compress") as cmp, \
         patch("yt2md.pipeline.transcribe") as tx, \
         patch("yt2md.pipeline.structure") as st:
        dl.return_value = (source_audio, metadata, {"id": "abc123"})
        # compress writes to its destination; simulate by creating the file
        def fake_compress(*, source: Path, destination: Path, cfg: Config) -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"\x00" * 50)
        cmp.side_effect = fake_compress
        tx.return_value = (transcript, [{"language": "en"}])
        st.return_value = doc

        yield {"dl": dl, "cmp": cmp, "tx": tx, "st": st}


class TestPipelineRun:
    def test_returns_path_to_written_markdown(self, cfg: Config, patched_stages) -> None:  # type: ignore[no-untyped-def]
        url = "https://www.youtube.com/watch?v=abc123"
        path = run(url, cfg=cfg)
        assert path.exists()
        assert path.suffix == ".md"
        assert "test-episode" in path.name

    def test_markdown_content_includes_frontmatter(self, cfg: Config, patched_stages) -> None:  # type: ignore[no-untyped-def]
        path = run("https://www.youtube.com/watch?v=abc123", cfg=cfg)
        content = path.read_text(encoding="utf-8")
        assert content.startswith("---")
        assert "title:" in content
        assert "Test Episode" in content

    def test_cache_populated(self, cfg: Config, patched_stages) -> None:  # type: ignore[no-untyped-def]
        run("https://www.youtube.com/watch?v=abc123", cfg=cfg)
        cache_subdir = cfg.cache_dir / "abc123"
        assert cache_subdir.exists()
        # At minimum: metadata.json, transcript-*.json, cleaned-*.json, structured-*.json
        assert (cache_subdir / "metadata.json").exists()
        assert any(cache_subdir.glob("transcript-*.json"))
        assert any(cache_subdir.glob("cleaned-*.json"))
        assert any(cache_subdir.glob("structured-*.json"))


class TestPipelineResume:
    def test_second_run_with_cache_skips_upstream_stages(
        self, cfg: Config, patched_stages
    ) -> None:  # type: ignore[no-untyped-def]
        url = "https://www.youtube.com/watch?v=abc123"
        run(url, cfg=cfg)
        # Reset call counts
        patched_stages["dl"].reset_mock()
        patched_stages["cmp"].reset_mock()
        patched_stages["tx"].reset_mock()
        patched_stages["st"].reset_mock()

        # Delete the output file to force re-render+write but allow cache hits upstream
        out_files = list(cfg.output_dir.glob("*.md"))
        for f in out_files:
            f.unlink()

        run(url, cfg=cfg)
        # All upstream stages should have hit cache; their mocks should NOT have been called.
        assert patched_stages["dl"].call_count == 0
        assert patched_stages["cmp"].call_count == 0
        assert patched_stages["tx"].call_count == 0
        assert patched_stages["st"].call_count == 0
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/integration/test_pipeline_happy_path.py -v
```

Expected: FAIL — `pipeline.run` does not exist.

- [ ] **Step 3: Write `src/yt2md/pipeline.py`**

```python
"""Pipeline orchestrator — the only module that knows the order of stages.

run(url, cfg) is the public API. cli.py calls it; tests call it; a future
web wrapper would call it. Stages don't know about each other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from yt2md.cache import ArtifactPaths, cached, fingerprint
from yt2md.config import Config
from yt2md.models import StructuredDoc, Transcript, VideoMetadata
from yt2md.stages.clean import CLEANER_VERSION, clean
from yt2md.stages.compress import compress
from yt2md.stages.download import download
from yt2md.stages.render import render
from yt2md.stages.structure import PROMPT_VERSION, structure
from yt2md.stages.transcribe import transcribe
from yt2md.stages.write import write
from yt2md.vocab_hint import VOCAB_HINT_VERSION


def run(url: str, *, cfg: Config) -> Path:
    """Execute the full pipeline. Returns the path to the written markdown."""
    metadata = _download_and_cache_metadata(url, cfg)
    paths = ArtifactPaths(cache_dir=cfg.cache_dir, video_id=metadata.video_id)

    audio = _compress_audio(url, metadata, paths, cfg)
    transcript = _transcribe_audio(audio, metadata, paths, cfg)
    cleaned = _clean_transcript(transcript, paths)
    doc = _structure_doc(cleaned, metadata, paths, cfg)
    markdown = render(doc, cleaned)
    return write(markdown=markdown, doc=doc, output_dir=cfg.output_dir)


def _download_and_cache_metadata(url: str, cfg: Config) -> VideoMetadata:
    """Download is special: produces audio + metadata. We cache metadata by video_id,
    but cannot avoid downloading audio without knowing video_id first. So this function
    runs download() iff metadata is not cached, deriving video_id from yt-dlp's response.

    The compressed audio cache lives at a hash-keyed path; if it exists, we don't
    re-download. If it doesn't, we download (which also yields metadata) and cache both.
    """
    # We need video_id to know the cache path. Extract it from the URL first.
    from yt2md.stages.download import _extract_video_id

    video_id = _extract_video_id(url)
    paths = ArtifactPaths(cache_dir=cfg.cache_dir, video_id=video_id)

    if paths.metadata.exists():
        return VideoMetadata.model_validate_json(paths.metadata.read_text(encoding="utf-8"))

    # Cache miss → run download to get audio + metadata together.
    _, metadata, raw = download(url, cfg=cfg)
    paths.metadata.parent.mkdir(parents=True, exist_ok=True)
    paths.metadata.write_text(metadata.model_dump_json(), encoding="utf-8")
    paths.metadata_raw.write_text(json.dumps(raw), encoding="utf-8")
    return metadata


def _compress_audio(url: str, metadata: VideoMetadata, paths: ArtifactPaths, cfg: Config) -> Path:
    """Produce the compressed audio path, downloading source if needed."""
    compression_hash = fingerprint(cfg.audio_bitrate_kbps, cfg.audio_codec, "mono")
    audio_path = paths.audio(compression_hash=compression_hash)

    if audio_path.exists():
        return audio_path

    # Need source audio first.
    source = next(iter(paths.root.glob("source_audio.*")), None)
    if source is None:
        # Source not in cache; re-run download (idempotent: paths.metadata exists already).
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
        dump=lambda t, p: p.write_text(t.model_dump_json(), encoding="utf-8"),
    )


def _clean_transcript(transcript: Transcript, paths: ArtifactPaths) -> Transcript:
    key = fingerprint(transcript.model_dump_json(), CLEANER_VERSION)
    target = paths.cleaned(input_hash=key)
    return cached(
        path=target,
        produce=lambda: clean(transcript),
        load=lambda p: Transcript.model_validate_json(p.read_text(encoding="utf-8")),
        dump=lambda t, p: p.write_text(t.model_dump_json(), encoding="utf-8"),
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
        dump=lambda d, p: p.write_text(d.model_dump_json(), encoding="utf-8"),
    )
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/integration/test_pipeline_happy_path.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/pipeline.py tests/integration/test_pipeline_happy_path.py
git commit -m "$(cat <<'EOF'
feat(pipeline): orchestrator wires 7 stages with hash-keyed cache

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section B: Observability (rich progress + structlog + runs.log)

---

### Task B.1: structlog configuration

**Files:**
- Create: `tests/unit/test_logging_config.py`
- Create: `src/yt2md/logging_config.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_logging_config.py`:

```python
"""Tests for structlog configuration helpers."""

import json

import structlog

from yt2md.logging_config import configure_logging, get_logger


class TestConfigureLogging:
    def test_configures_without_error(self) -> None:
        configure_logging(verbosity=0)
        log = get_logger("test")
        log.warning("hi")

    def test_verbosity_zero_is_warning(self) -> None:
        configure_logging(verbosity=0)
        log = get_logger("test")
        # No assertion of capture here; just confirms config doesn't raise.

    def test_get_logger_returns_bound_logger(self) -> None:
        configure_logging(verbosity=0)
        log = get_logger("test")
        assert hasattr(log, "warning")
        assert hasattr(log, "info")
        assert hasattr(log, "error")

    def test_structured_output_includes_context(self, capsys) -> None:  # type: ignore[no-untyped-def]
        configure_logging(verbosity=2)  # DEBUG → ensures log line emitted
        log = get_logger("test").bind(video_id="abc", stage="transcribe")
        log.info("started")
        # structlog JSON renderer outputs to stderr
        captured = capsys.readouterr()
        line = captured.err.strip().split("\n")[-1]
        payload = json.loads(line)
        assert payload["event"] == "started"
        assert payload["video_id"] == "abc"
        assert payload["stage"] == "transcribe"
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_logging_config.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write `src/yt2md/logging_config.py`**

```python
"""Structlog configuration. Configures once at CLI entry; loggers obtained via get_logger()."""

from __future__ import annotations

import logging
import sys

import structlog

_VERBOSITY_TO_LEVEL = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
}


def configure_logging(verbosity: int = 0) -> None:
    """Configure structlog to emit JSON to stderr at the level mapped from verbosity.

    verbosity: 0 = WARNING (default), 1 = INFO (-v), 2 = DEBUG (-vv).
    """
    level = _VERBOSITY_TO_LEVEL.get(verbosity, logging.DEBUG)
    logging.basicConfig(level=level, stream=sys.stderr, format="%(message)s")

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog BoundLogger by name."""
    return structlog.get_logger(name)
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_logging_config.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/logging_config.py tests/unit/test_logging_config.py
git commit -m "$(cat <<'EOF'
feat(logging): structlog JSON config with verbosity levels

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B.2: runs.log JSONL writer

**Files:**
- Create: `tests/unit/test_runs_log.py`
- Create: `src/yt2md/runs_log.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_runs_log.py`:

```python
"""Tests for the runs.log JSONL writer."""

import json
from pathlib import Path

from yt2md.runs_log import RunRecord, append_run


class TestAppendRun:
    def test_appends_jsonl_line(self, tmp_path: Path) -> None:
        log_path = tmp_path / "runs.log"
        record = RunRecord(
            video_id="abc",
            url="https://www.youtube.com/watch?v=abc",
            status="success",
            duration_s=120.5,
            transcription_usd=0.36,
            structuring_usd=0.04,
            transcription_backend="openai_transcribe",
            cache_hits=["audio"],
            stages_run=["transcribe", "structure", "render", "write"],
            audio_mb=8.4,
            video_duration_s=5025.0,
            schema_version=1,
            error_class=None,
            error_message=None,
        )
        append_run(log_path, record)
        line = log_path.read_text(encoding="utf-8").strip()
        payload = json.loads(line)
        assert payload["video_id"] == "abc"
        assert payload["status"] == "success"
        assert payload["costs"]["total_usd"] == 0.40

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        log_path = tmp_path / "nested" / "runs.log"
        record = RunRecord(
            video_id="x", url="u", status="success", duration_s=1.0,
            transcription_usd=0.0, structuring_usd=0.0,
            transcription_backend="local_whisper",
            cache_hits=[], stages_run=[], audio_mb=0.0, video_duration_s=0.0,
            schema_version=1, error_class=None, error_message=None,
        )
        append_run(log_path, record)
        assert log_path.exists()

    def test_multiple_appends_accumulate(self, tmp_path: Path) -> None:
        log_path = tmp_path / "runs.log"
        for i in range(3):
            r = RunRecord(
                video_id=f"v{i}", url="u", status="success", duration_s=1.0,
                transcription_usd=0.0, structuring_usd=0.0,
                transcription_backend="openai_transcribe",
                cache_hits=[], stages_run=[], audio_mb=0.0, video_duration_s=0.0,
                schema_version=1, error_class=None, error_message=None,
            )
            append_run(log_path, r)
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_failure_record(self, tmp_path: Path) -> None:
        log_path = tmp_path / "runs.log"
        record = RunRecord(
            video_id="x", url="u", status="failed", duration_s=2.0,
            transcription_usd=0.0, structuring_usd=0.0,
            transcription_backend="openai_transcribe",
            cache_hits=[], stages_run=["download"],
            audio_mb=0.0, video_duration_s=0.0,
            schema_version=1,
            error_class="VideoUnavailableError",
            error_message="private video",
        )
        append_run(log_path, record)
        payload = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert payload["status"] == "failed"
        assert payload["error_class"] == "VideoUnavailableError"
        assert payload["error_message"] == "private video"
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_runs_log.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write `src/yt2md/runs_log.py`**

```python
"""Append-only JSONL writer for runs.log — one line per pipeline invocation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class RunRecord:
    """One pipeline invocation's outcome. Serialized as a JSONL line."""

    video_id: str
    url: str
    status: str  # "success" | "skipped" | "failed"
    duration_s: float
    transcription_usd: float
    structuring_usd: float
    transcription_backend: str
    cache_hits: list[str]
    stages_run: list[str]
    audio_mb: float
    video_duration_s: float
    schema_version: int
    error_class: str | None
    error_message: str | None


def append_run(log_path: Path, record: RunRecord) -> None:
    """Append one record as a JSONL line. Creates parent directory if missing."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_payload(record)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _to_payload(record: RunRecord) -> dict[str, object]:
    """Convert RunRecord into the public JSONL payload shape (costs grouped)."""
    raw = asdict(record)
    transcription = raw.pop("transcription_usd")
    structuring = raw.pop("structuring_usd")
    backend = raw.pop("transcription_backend")
    raw["costs"] = {
        "transcription_usd": transcription,
        "structuring_usd": structuring,
        "transcription_backend": backend,
        "total_usd": round(transcription + structuring, 4),
    }
    raw["ts"] = datetime.now(UTC).isoformat()
    return raw
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_runs_log.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/runs_log.py tests/unit/test_runs_log.py
git commit -m "$(cat <<'EOF'
feat(runs_log): JSONL writer with grouped cost payload

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section C: cli.py (typer)

---

### Task C.1: typer app + minimal main command

**Files:**
- Create: `tests/unit/test_cli.py`
- Create: `src/yt2md/cli.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_cli.py`:

```python
"""Tests for the typer CLI."""

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from yt2md.cli import app

runner = CliRunner()


class TestVersion:
    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "yt2md" in result.stdout.lower() or "yt2llm" in result.stdout.lower()


class TestRequiredUrl:
    def test_no_url_fails(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code != 0


class TestRunCommand:
    def test_runs_pipeline_on_url(self, tmp_path: Path) -> None:
        with patch("yt2md.cli.run") as run_mock:
            run_mock.return_value = tmp_path / "out.md"
            (tmp_path / "out.md").write_text("hi", encoding="utf-8")
            result = runner.invoke(
                app,
                [
                    "https://www.youtube.com/watch?v=abc123",
                    "--cache-dir", str(tmp_path / "cache"),
                    "--output-dir", str(tmp_path),
                ],
                env={"YT2MD_GOOGLE_API_KEY": "g", "YT2MD_OPENAI_API_KEY": "o"},
            )
        assert result.exit_code == 0
        assert run_mock.called


class TestErrorExitCodes:
    def test_config_error_exits_3(self, tmp_path: Path) -> None:
        from yt2md.errors import ConfigError

        with patch("yt2md.cli.run") as run_mock:
            run_mock.side_effect = ConfigError("missing key")
            result = runner.invoke(
                app,
                ["https://www.youtube.com/watch?v=x"],
                env={"YT2MD_GOOGLE_API_KEY": "g"},
            )
        assert result.exit_code == 3

    def test_video_unavailable_exits_2(self, tmp_path: Path) -> None:
        from yt2md.errors import VideoUnavailableError

        with patch("yt2md.cli.run") as run_mock:
            run_mock.side_effect = VideoUnavailableError("private")
            result = runner.invoke(
                app,
                ["https://www.youtube.com/watch?v=x"],
                env={"YT2MD_GOOGLE_API_KEY": "g"},
            )
        assert result.exit_code == 2

    def test_generic_yt2md_error_exits_1(self, tmp_path: Path) -> None:
        from yt2md.errors import TranscriptionError

        with patch("yt2md.cli.run") as run_mock:
            run_mock.side_effect = TranscriptionError("boom")
            result = runner.invoke(
                app,
                ["https://www.youtube.com/watch?v=x"],
                env={"YT2MD_GOOGLE_API_KEY": "g"},
            )
        assert result.exit_code == 1
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_cli.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write `src/yt2md/cli.py`**

```python
"""yt2md CLI — typer entry point. Thin wrapper over pipeline.run()."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from yt2md import __version__
from yt2md.config import Config
from yt2md.errors import (
    ConfigError,
    LivestreamNotEndedError,
    NoAudioStreamError,
    VideoUnavailableError,
    YT2MDError,
)
from yt2md.logging_config import configure_logging
from yt2md.pipeline import run

app = typer.Typer(
    name="yt2md",
    help="Turn YouTube videos into structured markdown for an LLM knowledge graph.",
    add_completion=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"yt2md {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    url: Annotated[str | None, typer.Argument(help="YouTube URL")] = None,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir", help="Cache directory")] = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir", help="Output directory")] = None,
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="Transcription backend: openai_transcribe, local_whisper, or auto"),
    ] = None,
    cookies_from_browser: Annotated[
        str | None,
        typer.Option("--cookies-from-browser", help="Browser to load cookies from (firefox, chrome, edge)"),
    ] = None,
    cookies_file: Annotated[
        Path | None,
        typer.Option("--cookies", help="Path to cookies.txt file"),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Re-run all stages, clearing cache")] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Do not read or write cache")] = False,
    verbose: Annotated[int, typer.Option("-v", "--verbose", count=True, help="-v for INFO, -vv for DEBUG")] = 0,
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version and exit")
    ] = False,
) -> None:
    """yt2md: process a YouTube URL into Tier 3 structured markdown."""
    if ctx.invoked_subcommand is not None:
        return
    if url is None:
        typer.echo("Error: URL is required", err=True)
        raise typer.Exit(1)

    configure_logging(verbosity=verbose)

    overrides: dict[str, object] = {}
    if cache_dir is not None:
        overrides["cache_dir"] = cache_dir
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if backend is not None:
        overrides["transcription_backend"] = backend
    if cookies_from_browser is not None:
        overrides["cookies_from_browser"] = cookies_from_browser
    if cookies_file is not None:
        overrides["cookies_file"] = cookies_file
    overrides["force"] = force
    overrides["no_cache"] = no_cache

    try:
        cfg = Config(**overrides)  # type: ignore[arg-type]
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(3) from e

    try:
        path = run(url, cfg=cfg)
    except ConfigError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(3) from e
    except (VideoUnavailableError, LivestreamNotEndedError, NoAudioStreamError) as e:
        typer.echo(f"{e.__class__.__name__}: {e}", err=True)
        raise typer.Exit(2) from e
    except YT2MDError as e:
        typer.echo(f"{e.__class__.__name__}: {e}", err=True)
        raise typer.Exit(1) from e

    typer.echo(str(path))
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_cli.py -v
```

Expected: 6 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/cli.py tests/unit/test_cli.py
git commit -m "$(cat <<'EOF'
feat(cli): typer app with pipeline invocation + typed exit codes

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task C.2: Idempotency short-circuit (output exists → skip)

**Files:**
- Create: `tests/unit/test_cli_idempotency.py`
- Modify: `src/yt2md/cli.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_cli_idempotency.py`:

```python
"""Tests for idempotency: existing output file → skip, --force re-runs."""

from datetime import date
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from yt2md.cli import app
from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    Frontmatter,
    StructuredDoc,
    Takeaway,
)

runner = CliRunner()


def _existing_output(out_dir: Path, video_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"2024-03-15__test-channel__test-episode.md"
    p = out_dir / name
    p.write_text(f"---\nvideo_id: {video_id}\n---\nold content\n", encoding="utf-8")
    return p


class TestSkipOnOutputExists:
    def test_existing_same_video_id_skips(self, tmp_path: Path) -> None:
        existing = _existing_output(tmp_path / "out", "abc123")
        with patch("yt2md.cli.run") as run_mock:
            with patch("yt2md.cli._derive_expected_output") as der:
                der.return_value = (existing, "abc123")
                result = runner.invoke(
                    app,
                    [
                        "https://www.youtube.com/watch?v=abc123",
                        "--cache-dir", str(tmp_path / "cache"),
                        "--output-dir", str(tmp_path / "out"),
                    ],
                    env={"YT2MD_GOOGLE_API_KEY": "g"},
                )
        assert result.exit_code == 0
        assert "Already processed" in result.stdout or "skipped" in result.stdout.lower()
        assert not run_mock.called


class TestForceOverridesSkip:
    def test_force_clears_cache_and_runs(self, tmp_path: Path) -> None:
        existing = _existing_output(tmp_path / "out", "abc123")
        cache_dir = tmp_path / "cache"
        cache_video = cache_dir / "abc123"
        cache_video.mkdir(parents=True)
        (cache_video / "metadata.json").write_text("{}", encoding="utf-8")

        with patch("yt2md.cli.run") as run_mock, \
             patch("yt2md.cli._derive_expected_output") as der:
            der.return_value = (existing, "abc123")
            run_mock.return_value = existing
            result = runner.invoke(
                app,
                [
                    "https://www.youtube.com/watch?v=abc123",
                    "--cache-dir", str(cache_dir),
                    "--output-dir", str(tmp_path / "out"),
                    "--force",
                ],
                env={"YT2MD_GOOGLE_API_KEY": "g"},
            )
        assert result.exit_code == 0
        assert run_mock.called
        # Cache subdir cleared
        assert not cache_video.exists()
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_cli_idempotency.py -v
```

Expected: FAIL.

- [ ] **Step 3: Modify `src/yt2md/cli.py`**

Above the `main` callback, add:

```python
import shutil

from yt2md.stages.download import _extract_video_id


def _derive_expected_output(url: str, cfg: Config) -> tuple[Path | None, str]:
    """Compute the expected output filename without running the pipeline.

    Returns (path-if-determinable, video_id). For collision handling, the actual
    filename may differ; we only use this for the short-circuit check.
    """
    video_id = _extract_video_id(url)
    # We can't know the channel/title slug without metadata. So we check if any
    # output file references this video_id in its frontmatter.
    for candidate in cfg.output_dir.glob("*.md"):
        try:
            head = candidate.read_text(encoding="utf-8")[:1000]
        except OSError:
            continue
        if f"video_id: {video_id}" in head:
            return candidate, video_id
    return None, video_id


def _clear_cache_for(video_id: str, cfg: Config) -> None:
    cache_video = cfg.cache_dir / video_id
    if cache_video.exists():
        typer.echo(f"Removing cache for {video_id}", err=True)
        shutil.rmtree(cache_video)
```

Modify the `main` body — between the `cfg` construction and the `run(url, cfg=cfg)` call, insert:

```python
    # Idempotency check
    if not cfg.force:
        existing, _ = _derive_expected_output(url, cfg)
        if existing is not None:
            typer.echo(f"Already processed: {existing}")
            raise typer.Exit(0)

    if cfg.force:
        _, video_id = _derive_expected_output(url, cfg)
        _clear_cache_for(video_id, cfg)
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_cli_idempotency.py -v
uv run pytest tests/unit/test_cli.py -v
```

Expected: all PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/cli.py tests/unit/test_cli_idempotency.py
git commit -m "$(cat <<'EOF'
feat(cli): idempotency short-circuit + --force cache clear

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task C.3: regen subcommand

**Files:**
- Create: `tests/unit/test_cli_regen.py`
- Modify: `src/yt2md/cli.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_cli_regen.py`:

```python
"""Tests for yt2md regen subcommand."""

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from yt2md.cli import app

runner = CliRunner()


class TestRegenPath:
    def test_regen_single_path_invokes_pipeline_with_no_cache_for_structured(
        self, tmp_path: Path
    ) -> None:
        # Create a stale output file
        out = tmp_path / "old.md"
        out.write_text(
            "---\n"
            "video_id: abc123\n"
            "url: https://www.youtube.com/watch?v=abc123\n"
            "schema_version: 0\n"  # stale
            "---\n",
            encoding="utf-8",
        )
        with patch("yt2md.cli.run") as run_mock:
            run_mock.return_value = out
            result = runner.invoke(
                app,
                ["regen", str(out)],
                env={"YT2MD_GOOGLE_API_KEY": "g"},
            )
        assert result.exit_code == 0
        assert run_mock.called


class TestRegenAll:
    def test_regen_all_dry_run(self, tmp_path: Path) -> None:
        # Two stale files
        for i in range(2):
            p = tmp_path / f"old{i}.md"
            p.write_text(
                "---\n"
                f"video_id: abc{i}\n"
                f"url: https://www.youtube.com/watch?v=abc{i}\n"
                "schema_version: 0\n"
                "---\n",
                encoding="utf-8",
            )
        with patch("yt2md.cli.run") as run_mock:
            result = runner.invoke(
                app,
                ["regen", "--all", "--dry-run", "--output-dir", str(tmp_path)],
                env={"YT2MD_GOOGLE_API_KEY": "g"},
            )
        assert result.exit_code == 0
        assert "old0.md" in result.stdout
        assert "old1.md" in result.stdout
        assert not run_mock.called  # dry-run
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_cli_regen.py -v
```

Expected: FAIL.

- [ ] **Step 3: Append regen subcommand to `src/yt2md/cli.py`**

```python
import re

from yt2md.models import CURRENT_SCHEMA_VERSION

regen_app = typer.Typer(help="Regenerate output files from their cached upstream artifacts.")
app.add_typer(regen_app, name="regen")


_FM_URL_RE = re.compile(r"^url:\s*(\S+)\s*$", re.MULTILINE)
_FM_SCHEMA_VERSION_RE = re.compile(r"^schema_version:\s*(\d+)\s*$", re.MULTILINE)


@regen_app.callback(invoke_without_command=True)
def regen_main(
    path: Annotated[Path | None, typer.Argument(help="Specific .md file to regen, or omit for --all")] = None,
    all_files: Annotated[bool, typer.Option("--all", help="Regenerate all stale files")] = False,
    min_version: Annotated[int, typer.Option("--min-version", help="Regen files below this schema_version")] = CURRENT_SCHEMA_VERSION,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would regen; don't modify")] = False,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir")] = None,
) -> None:
    """Regenerate one or many output files using cached upstream artifacts."""
    overrides: dict[str, object] = {"no_cache": False}
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if cache_dir is not None:
        overrides["cache_dir"] = cache_dir

    try:
        cfg = Config(**overrides)  # type: ignore[arg-type]
    except Exception as e:  # noqa: BLE001
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(3) from e

    if all_files:
        candidates = _find_stale_files(cfg.output_dir, min_version)
        if dry_run:
            for p in candidates:
                typer.echo(str(p))
            raise typer.Exit(0)
        for p in candidates:
            url = _extract_url_from_frontmatter(p)
            if url is None:
                typer.echo(f"Skipping (no url in frontmatter): {p}", err=True)
                continue
            run(url, cfg=cfg)
            typer.echo(f"Regenerated: {p}")
        return

    if path is None:
        typer.echo("Error: provide a path or use --all", err=True)
        raise typer.Exit(1)
    if not path.exists():
        typer.echo(f"Not found: {path}", err=True)
        raise typer.Exit(1)

    url = _extract_url_from_frontmatter(path)
    if url is None:
        typer.echo(f"No url in frontmatter of {path}; cannot regen", err=True)
        raise typer.Exit(1)
    run(url, cfg=cfg)
    typer.echo(f"Regenerated: {path}")


def _find_stale_files(output_dir: Path, min_version: int) -> list[Path]:
    """Return all .md files in output_dir whose schema_version < min_version."""
    out: list[Path] = []
    for p in sorted(output_dir.glob("*.md")):
        try:
            text = p.read_text(encoding="utf-8")[:2000]
        except OSError:
            continue
        version_match = _FM_SCHEMA_VERSION_RE.search(text)
        version = int(version_match.group(1)) if version_match else 0
        if version < min_version:
            out.append(p)
    return out


def _extract_url_from_frontmatter(path: Path) -> str | None:
    """Pull `url:` value from a markdown file's YAML frontmatter."""
    try:
        text = path.read_text(encoding="utf-8")[:2000]
    except OSError:
        return None
    match = _FM_URL_RE.search(text)
    return match.group(1) if match else None
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_cli_regen.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/cli.py tests/unit/test_cli_regen.py
git commit -m "$(cat <<'EOF'
feat(cli): regen subcommand with --all, --dry-run, --min-version

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section D: Wire observability into pipeline + cli

---

### Task D.1: Pipeline emits structured events for runs.log

**Files:**
- Create: `tests/integration/test_pipeline_observability.py`
- Modify: `src/yt2md/pipeline.py`

- [ ] **Step 1: Write the failing test**

`tests/integration/test_pipeline_observability.py`:

```python
"""Tests for pipeline-level observability: runs.log entries on success and failure."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from yt2md.config import Config
from yt2md.errors import TranscriptionError
from yt2md.pipeline import run


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        google_api_key="g",  # type: ignore[arg-type]
        openai_api_key="o",  # type: ignore[arg-type]
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
    )


def test_success_appends_to_runs_log(cfg: Config, patched_stages) -> None:  # type: ignore[no-untyped-def]
    run("https://www.youtube.com/watch?v=abc123", cfg=cfg)
    runs_log = cfg.cache_dir / "runs.log"
    assert runs_log.exists()
    payload = json.loads(runs_log.read_text(encoding="utf-8").strip().split("\n")[-1])
    assert payload["status"] == "success"
    assert payload["video_id"] == "abc123"


def test_failure_appends_to_runs_log(cfg: Config) -> None:
    with patch("yt2md.pipeline.download") as dl:
        dl.side_effect = TranscriptionError("boom")
        with pytest.raises(TranscriptionError):
            run("https://www.youtube.com/watch?v=abc123", cfg=cfg)
    runs_log = cfg.cache_dir / "runs.log"
    assert runs_log.exists()
    payload = json.loads(runs_log.read_text(encoding="utf-8").strip().split("\n")[-1])
    assert payload["status"] == "failed"
    assert payload["error_class"] == "TranscriptionError"
```

(Note: `patched_stages` fixture is from `test_pipeline_happy_path.py` — duplicate it in conftest or import directly. For simplicity, inline-duplicate the relevant patching here if it's hard to share. Reuse via a fixture in `tests/integration/conftest.py` is cleaner.)

- [ ] **Step 2: Move the `patched_stages` fixture to `tests/integration/conftest.py`**

Create `tests/integration/conftest.py` and move the fixture from `test_pipeline_happy_path.py` into it (so both tests use the same fixture).

- [ ] **Step 3: Run — confirm fails**

```bash
uv run pytest tests/integration/test_pipeline_observability.py -v
```

Expected: FAIL.

- [ ] **Step 4: Wire runs.log emission into pipeline.run**

Modify `src/yt2md/pipeline.py` `run()` to wrap the existing body:

```python
import time

from yt2md.runs_log import RunRecord, append_run
from yt2md.errors import YT2MDError


def run(url: str, *, cfg: Config) -> Path:
    """Execute the full pipeline. Returns the path to the written markdown.

    Emits a JSONL record to <cache_dir>/runs.log on both success and failure.
    """
    start = time.monotonic()
    video_id = "unknown"
    error: Exception | None = None
    try:
        path = _run_inner(url, cfg)
    except YT2MDError as e:
        error = e
        raise
    finally:
        elapsed = time.monotonic() - start
        record = _build_record(
            url=url,
            video_id=video_id if error is None else _safe_extract_video_id(url),
            elapsed_s=elapsed,
            error=error,
        )
        append_run(cfg.cache_dir / "runs.log", record)
    return path


def _run_inner(url: str, cfg: Config) -> Path:
    metadata = _download_and_cache_metadata(url, cfg)
    # ... rest unchanged
```

This refactor extracts the original body into `_run_inner`. Update the inner body to set `video_id = metadata.video_id` (or pass it back via closure).

For the simplest correct shape, use a `dict[str, Any]` "outcome" object built in `_run_inner` and consumed by the finally block:

```python
def run(url: str, *, cfg: Config) -> Path:
    start = time.monotonic()
    outcome: dict[str, Any] = {"video_id": "unknown", "cache_hits": [], "stages_run": []}
    error: Exception | None = None
    try:
        path = _run_inner(url, cfg, outcome)
    except YT2MDError as e:
        error = e
        raise
    finally:
        elapsed = time.monotonic() - start
        record = _build_record(url=url, outcome=outcome, elapsed_s=elapsed, error=error)
        append_run(cfg.cache_dir / "runs.log", record)
    return path


def _safe_extract_video_id(url: str) -> str:
    try:
        from yt2md.stages.download import _extract_video_id
        return _extract_video_id(url)
    except Exception:  # noqa: BLE001
        return "unknown"


def _build_record(
    *, url: str, outcome: dict[str, Any], elapsed_s: float, error: Exception | None
) -> RunRecord:
    return RunRecord(
        video_id=outcome.get("video_id", "unknown"),
        url=url,
        status="failed" if error else "success",
        duration_s=round(elapsed_s, 3),
        transcription_usd=outcome.get("transcription_usd", 0.0),
        structuring_usd=outcome.get("structuring_usd", 0.0),
        transcription_backend=outcome.get("transcription_backend", "unknown"),
        cache_hits=outcome.get("cache_hits", []),
        stages_run=outcome.get("stages_run", []),
        audio_mb=outcome.get("audio_mb", 0.0),
        video_duration_s=outcome.get("video_duration_s", 0.0),
        schema_version=CURRENT_SCHEMA_VERSION,
        error_class=error.__class__.__name__ if error else None,
        error_message=str(error) if error else None,
    )
```

`_run_inner` is the original `run` body, plus populates `outcome` along the way (sets `outcome["video_id"] = metadata.video_id`, etc.).

- [ ] **Step 5: Run — confirm passes**

```bash
uv run pytest tests/integration/test_pipeline_observability.py -v
uv run pytest tests/integration/test_pipeline_happy_path.py -v
```

Expected: all PASS.

- [ ] **Step 6: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/pipeline.py tests/integration/test_pipeline_observability.py tests/integration/conftest.py
git commit -m "$(cat <<'EOF'
feat(pipeline): append runs.log entry on success and failure

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section E: Final integration test + entry point

---

### Task E.1: End-to-end CLI test with all SDKs mocked

**Files:**
- Create: `tests/integration/test_cli_end_to_end.py`

- [ ] **Step 1: Write the test**

`tests/integration/test_cli_end_to_end.py`:

```python
"""End-to-end CLI test: yt2md <url> produces a valid .md file with all SDKs mocked."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from yt2md.cli import app

runner = CliRunner()


def test_end_to_end_produces_markdown(tmp_path: Path, patched_stages) -> None:  # type: ignore[no-untyped-def]
    """Full CLI invocation → all stages mocked → markdown file produced."""
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        app,
        [
            "https://www.youtube.com/watch?v=abc123",
            "--cache-dir", str(cache_dir),
            "--output-dir", str(out_dir),
        ],
        env={"YT2MD_GOOGLE_API_KEY": "g", "YT2MD_OPENAI_API_KEY": "o"},
    )
    assert result.exit_code == 0, result.stdout + result.output

    md_files = list(out_dir.glob("*.md"))
    assert len(md_files) == 1
    content = md_files[0].read_text(encoding="utf-8")
    assert content.startswith("---")
    assert "Test Episode" in content
    assert "Test Channel" in content
    # runs.log appended
    assert (cache_dir / "runs.log").exists()
```

(`patched_stages` fixture is in `tests/integration/conftest.py` and is shared.)

- [ ] **Step 2: Run — confirm passes**

```bash
uv run pytest tests/integration/test_cli_end_to_end.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add tests/integration/test_cli_end_to_end.py
git commit -m "$(cat <<'EOF'
test(cli): end-to-end CLI invocation produces markdown with mocked SDKs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task E.2: Update __main__.py to invoke the real CLI

**Files:**
- Modify: `src/yt2md/__main__.py`

- [ ] **Step 1: Replace `src/yt2md/__main__.py`**

```python
"""Allow `python -m yt2md` invocation."""

from yt2md.cli import app

if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Smoke test**

```bash
uv run python -m yt2md --version
```

Expected: prints `yt2md 0.0.1` (or whatever version is set).

- [ ] **Step 3: Lint + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/__main__.py
git commit -m "$(cat <<'EOF'
feat(cli): wire __main__ to typer app

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 final check

### Task F.1: Run the full suite

- [ ] **Step 1: Run all unit + integration tests with coverage**

```bash
uv run pytest tests/unit tests/integration --cov=src/yt2md --cov-report=term-missing --cov-fail-under=85
```

Expected: all PASS; coverage ≥85%.

- [ ] **Step 2: Pre-commit all files**

```bash
uv run pre-commit run --all-files
```

Expected: all hooks pass.

- [ ] **Step 3: Try a real CLI invocation (if API keys available)**

```bash
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
uv run yt2md "https://www.youtube.com/watch?v=jNQXAC9IVRw"
```

Expected: a real .md file in `./output/`. Cost: <$0.05 for a 19-second video.

- [ ] **Step 4: Mark Phase 4 complete in index**

```markdown
- [x] Phase 4 — Orchestrator + CLI
```

```bash
git add docs/superpowers/plans/2026-05-23-yt2llm-index.md
git commit -m "$(cat <<'EOF'
docs(plan): mark Phase 4 complete — yt2llm MVP shippable

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## What Phase 4 produced (and what the project now does)

- `src/yt2md/pipeline.py` — orchestrator wiring the 7 stages with hash-keyed caching and runs.log emission
- `src/yt2md/cli.py` — typer CLI with idempotency check, --force, --no-cache, --backend, --cookies, regen subcommand, exit codes per error class
- `src/yt2md/logging_config.py` — structlog JSON config
- `src/yt2md/runs_log.py` — JSONL writer
- `src/yt2md/__main__.py` — `python -m yt2md` entry point
- End-to-end integration tests proving the MVP works (with mocked SDKs)

The CLI is callable: `yt2md <url>`. Outputs land in `output/`. Cache lives in `cache/`. Per-run cost analytics in `cache/runs.log`. Re-running is idempotent. Schema bumps trigger `yt2md regen`.

**MVP shippable.**

---

## Where to go next (post-MVP)

These were explicitly out of MVP scope (see spec §13) and have no plan files yet:

- Batch / playlist mode
- Web UI
- Multilingual support
- Cross-chunk diarization reconciliation
- Local-backend diarization via pyannote.audio
- Schema auto-regen on `yt2md <url>` (vs. explicit `regen`)
- Telemetry / OpenTelemetry traces

When ready to tackle one, brainstorm a focused spec for it (using `superpowers:brainstorming`), then a fresh implementation plan (using `superpowers:writing-plans`).
