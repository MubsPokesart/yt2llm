"""Tests for pipeline-level observability: runs.log entries on success and failure."""

import json
from typing import Any
from unittest.mock import patch

import pytest

from yt2md.config import Config
from yt2md.errors import TranscriptionError
from yt2md.pipeline import run


def test_success_appends_to_runs_log(cfg: Config, patched_stages: Any) -> None:  # noqa: ARG001
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
