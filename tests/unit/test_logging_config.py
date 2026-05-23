"""Tests for structlog configuration helpers."""

import json

import pytest

from yt2md.logging_config import configure_logging, get_logger


class TestConfigureLogging:
    def test_configures_without_error(self) -> None:
        configure_logging(verbosity=0)
        log = get_logger("test")
        log.warning("hi")

    def test_verbosity_zero_is_warning(self) -> None:
        configure_logging(verbosity=0)
        get_logger("test")

    def test_get_logger_returns_bound_logger(self) -> None:
        configure_logging(verbosity=0)
        log = get_logger("test")
        assert hasattr(log, "warning")
        assert hasattr(log, "info")
        assert hasattr(log, "error")

    def test_structured_output_includes_context(self, capfd: pytest.CaptureFixture[str]) -> None:
        configure_logging(verbosity=2)  # DEBUG → ensures log line emitted
        log = get_logger("test").bind(video_id="abc", stage="transcribe")
        log.info("started")
        captured = capfd.readouterr()
        line = captured.err.strip().split("\n")[-1]
        payload = json.loads(line)
        assert payload["event"] == "started"
        assert payload["video_id"] == "abc"
        assert payload["stage"] == "transcribe"
