"""Tests for the typer CLI."""

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from yt2md.cli import _inject_default_subcommand, app  # noqa: PLC2701  -- testing helper
from yt2md.errors import ConfigError, TranscriptionError, VideoUnavailableError

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

    def test_unknown_option_fails_loudly(self) -> None:
        """Single-dash typo of --force used to be silently absorbed by the
        DefaultCommandGroup hack. With the explicit `run` subcommand it must error.
        """
        result = runner.invoke(
            app,
            ["run", "https://www.youtube.com/watch?v=abc", "-force"],
            env={"YT2MD_GOOGLE_API_KEY": "g"},
        )
        assert result.exit_code != 0


class TestRunCommand:
    def test_runs_pipeline_on_url(self, tmp_path: Path) -> None:
        with patch("yt2md.cli.run") as run_mock:
            run_mock.return_value = tmp_path / "out.md"
            (tmp_path / "out.md").write_text("hi", encoding="utf-8")
            result = runner.invoke(
                app,
                [
                    "run",
                    "https://www.youtube.com/watch?v=abc123",
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--output-dir",
                    str(tmp_path),
                ],
                env={"YT2MD_GOOGLE_API_KEY": "g", "YT2MD_OPENAI_API_KEY": "o"},
            )
        assert result.exit_code == 0
        assert run_mock.called


class TestErrorExitCodes:
    def test_config_error_exits_3(self, tmp_path: Path) -> None:
        with patch("yt2md.cli.run") as run_mock:
            run_mock.side_effect = ConfigError("missing key")
            result = runner.invoke(
                app,
                ["run", "https://www.youtube.com/watch?v=x"],
                env={"YT2MD_GOOGLE_API_KEY": "g"},
            )
        assert result.exit_code == 3

    def test_video_unavailable_exits_2(self, tmp_path: Path) -> None:
        with patch("yt2md.cli.run") as run_mock:
            run_mock.side_effect = VideoUnavailableError("private")
            result = runner.invoke(
                app,
                ["run", "https://www.youtube.com/watch?v=x"],
                env={"YT2MD_GOOGLE_API_KEY": "g"},
            )
        assert result.exit_code == 2

    def test_generic_yt2md_error_exits_1(self, tmp_path: Path) -> None:
        with patch("yt2md.cli.run") as run_mock:
            run_mock.side_effect = TranscriptionError("boom")
            result = runner.invoke(
                app,
                ["run", "https://www.youtube.com/watch?v=x"],
                env={"YT2MD_GOOGLE_API_KEY": "g"},
            )
        assert result.exit_code == 1


class TestDefaultSubcommandInjection:
    """`yt2md <URL> ...` must keep working at the entrypoint level via argv injection."""

    def test_injects_run_when_first_positional_is_not_subcommand(self) -> None:
        assert _inject_default_subcommand(["https://youtu.be/x"]) == [
            "run",
            "https://youtu.be/x",
        ]
        assert _inject_default_subcommand(["https://youtu.be/x", "-v"]) == [
            "run",
            "https://youtu.be/x",
            "-v",
        ]

    def test_does_not_inject_when_subcommand_present(self) -> None:
        assert _inject_default_subcommand(["run", "https://youtu.be/x"]) == [
            "run",
            "https://youtu.be/x",
        ]
        assert _inject_default_subcommand(["regen", "--all"]) == ["regen", "--all"]

    def test_does_not_inject_for_top_level_flags(self) -> None:
        assert _inject_default_subcommand(["--version"]) == ["--version"]
        assert _inject_default_subcommand(["--help"]) == ["--help"]

    def test_empty_argv_unchanged(self) -> None:
        assert _inject_default_subcommand([]) == []
