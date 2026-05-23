"""Tests for yt2md regen subcommand."""

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from yt2md.cli import app

runner = CliRunner()


class TestRegenPath:
    def test_regen_single_path_invokes_pipeline(self, tmp_path: Path) -> None:
        out = tmp_path / "old.md"
        out.write_text(
            "---\n"
            "video_id: abc123\n"
            "url: https://www.youtube.com/watch?v=abc123\n"
            "schema_version: 0\n"
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
