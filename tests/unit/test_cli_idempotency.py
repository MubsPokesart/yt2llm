"""Tests for idempotency: existing output file → skip, --force re-runs."""

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from yt2md.cli import app

runner = CliRunner()


def _existing_output(out_dir: Path, video_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "2024-03-15__test-channel__test-episode.md"
    p = out_dir / name
    p.write_text(f"---\nvideo_id: {video_id}\n---\nold content\n", encoding="utf-8")
    return p


class TestSkipOnOutputExists:
    def test_existing_same_video_id_skips(self, tmp_path: Path) -> None:
        existing = _existing_output(tmp_path / "out", "abc123")
        with patch("yt2md.cli.run") as run_mock, patch("yt2md.cli._derive_expected_output") as der:
            der.return_value = (existing, "abc123")
            result = runner.invoke(
                app,
                [
                    "https://www.youtube.com/watch?v=abc123",
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--output-dir",
                    str(tmp_path / "out"),
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

        with patch("yt2md.cli.run") as run_mock, patch("yt2md.cli._derive_expected_output") as der:
            der.return_value = (existing, "abc123")
            run_mock.return_value = existing
            result = runner.invoke(
                app,
                [
                    "https://www.youtube.com/watch?v=abc123",
                    "--cache-dir",
                    str(cache_dir),
                    "--output-dir",
                    str(tmp_path / "out"),
                    "--force",
                ],
                env={"YT2MD_GOOGLE_API_KEY": "g"},
            )
        assert result.exit_code == 0
        assert run_mock.called
        assert not cache_video.exists()
