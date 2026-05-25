"""End-to-end CLI test: yt2md <url> produces a valid .md file with all SDKs mocked."""

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from yt2md.cli import app

runner = CliRunner()


def test_end_to_end_produces_markdown(
    tmp_path: Path,
    patched_stages: Any,  # noqa: ARG001 -- fixture is side-effecting
) -> None:
    """Full CLI invocation → all stages mocked → markdown file produced."""
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"

    result = runner.invoke(
        app,
        [
            "run",
            "https://www.youtube.com/watch?v=abc123",
            "--cache-dir",
            str(cache_dir),
            "--output-dir",
            str(out_dir),
        ],
        env={"YT2MD_GOOGLE_API_KEY": "g", "YT2MD_OPENAI_API_KEY": "o"},
    )
    assert result.exit_code == 0, result.stdout + (result.output or "")

    md_files = list(out_dir.glob("*.md"))
    assert len(md_files) == 1
    content = md_files[0].read_text(encoding="utf-8")
    assert content.startswith("---")
    assert "Test Episode" in content
    assert "Test Channel" in content
    # runs.log appended
    assert (cache_dir / "runs.log").exists()
