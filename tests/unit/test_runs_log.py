"""Tests for the runs.log JSONL writer."""

import json
from pathlib import Path

import pytest

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
        assert payload["costs"]["total_usd"] == pytest.approx(0.40)

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        log_path = tmp_path / "nested" / "runs.log"
        record = RunRecord(
            video_id="x",
            url="u",
            status="success",
            duration_s=1.0,
            transcription_usd=0.0,
            structuring_usd=0.0,
            transcription_backend="local_whisper",
            cache_hits=[],
            stages_run=[],
            audio_mb=0.0,
            video_duration_s=0.0,
            schema_version=1,
            error_class=None,
            error_message=None,
        )
        append_run(log_path, record)
        assert log_path.exists()

    def test_multiple_appends_accumulate(self, tmp_path: Path) -> None:
        log_path = tmp_path / "runs.log"
        for i in range(3):
            r = RunRecord(
                video_id=f"v{i}",
                url="u",
                status="success",
                duration_s=1.0,
                transcription_usd=0.0,
                structuring_usd=0.0,
                transcription_backend="openai_transcribe",
                cache_hits=[],
                stages_run=[],
                audio_mb=0.0,
                video_duration_s=0.0,
                schema_version=1,
                error_class=None,
                error_message=None,
            )
            append_run(log_path, r)
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_failure_record(self, tmp_path: Path) -> None:
        log_path = tmp_path / "runs.log"
        record = RunRecord(
            video_id="x",
            url="u",
            status="failed",
            duration_s=2.0,
            transcription_usd=0.0,
            structuring_usd=0.0,
            transcription_backend="openai_transcribe",
            cache_hits=[],
            stages_run=["download"],
            audio_mb=0.0,
            video_duration_s=0.0,
            schema_version=1,
            error_class="VideoUnavailableError",
            error_message="private video",
        )
        append_run(log_path, record)
        payload = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert payload["status"] == "failed"
        assert payload["error_class"] == "VideoUnavailableError"
        assert payload["error_message"] == "private video"
