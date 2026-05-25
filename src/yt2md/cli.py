"""yt2md CLI — typer entry point. Thin wrapper over pipeline.run().

The user-facing form `yt2md <URL>` is preserved by `main_entry`: it inspects sys.argv,
and if the first positional token is not a known subcommand it inserts `run` so the
URL goes through the `run` command. Inside the typer app there is no
default-command machinery — every invocation is an explicit subcommand, so unknown
options error loudly instead of being silently swallowed.
"""

from __future__ import annotations

import re
import shutil
import sys
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
from yt2md.models import CURRENT_SCHEMA_VERSION
from yt2md.pipeline import run
from yt2md.stages.download import _extract_video_id

FRONTMATTER_HEAD_BYTES = 2000

app = typer.Typer(
    name="yt2md",
    help="Turn YouTube videos into structured markdown for an LLM knowledge graph.",
    add_completion=False,
    no_args_is_help=True,
)


def _derive_expected_output(url: str, cfg: Config) -> tuple[Path | None, str]:
    """Compute the expected output filename without running the pipeline.

    Returns (path-if-determinable, video_id). For collision handling, the actual
    filename may differ; we only use this for the short-circuit check.
    """
    video_id = _extract_video_id(url)
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


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"yt2md {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = False,
) -> None:
    """yt2md — turn YouTube videos into structured markdown."""


def _build_run_overrides(
    cache_dir: Path | None,
    output_dir: Path | None,
    backend: str | None,
    cookies_from_browser: str | None,
    cookies_file: Path | None,
    force: bool,
    no_cache: bool,
) -> dict[str, object]:
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
    return overrides


@app.command("run")
def run_main(
    url: Annotated[str, typer.Argument(help="YouTube video URL")],
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir", help="Cache directory")] = None,
    output_dir: Annotated[
        Path | None, typer.Option("--output-dir", help="Output directory")
    ] = None,
    backend: Annotated[
        str | None,
        typer.Option(
            "--backend",
            help="Transcription backend: openai_transcribe, local_whisper, or auto",
        ),
    ] = None,
    cookies_from_browser: Annotated[
        str | None,
        typer.Option(
            "--cookies-from-browser",
            help="Browser to load cookies from (firefox, chrome, edge)",
        ),
    ] = None,
    cookies_file: Annotated[
        Path | None,
        typer.Option("--cookies", help="Path to cookies.txt file"),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-run all stages, clearing cache")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Do not read or write cache")
    ] = False,
    verbose: Annotated[
        int,
        typer.Option("-v", "--verbose", count=True, help="-v for INFO, -vv for DEBUG"),
    ] = 0,
) -> None:
    """Process a YouTube URL into Tier 3 structured markdown."""
    configure_logging(verbosity=verbose)
    overrides = _build_run_overrides(
        cache_dir, output_dir, backend, cookies_from_browser, cookies_file, force, no_cache
    )

    try:
        cfg = Config(**overrides)  # type: ignore[arg-type]
    except Exception as e:  # Config(**overrides) may raise any pydantic/validation error
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(3) from e

    if not cfg.force:
        existing, _ = _derive_expected_output(url, cfg)
        if existing is not None:
            typer.echo(f"Already processed: {existing}")
            raise typer.Exit(0)

    if cfg.force:
        _, video_id = _derive_expected_output(url, cfg)
        _clear_cache_for(video_id, cfg)

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


_FM_URL_RE = re.compile(r"^url:\s*(\S+)\s*$", re.MULTILINE)
_FM_SCHEMA_VERSION_RE = re.compile(r"^schema_version:\s*(\d+)\s*$", re.MULTILINE)


@app.command("regen")
def regen_main(
    path: Annotated[
        Path | None, typer.Argument(help="Specific .md file to regen, or omit for --all")
    ] = None,
    all_files: Annotated[bool, typer.Option("--all", help="Regenerate all stale files")] = False,
    min_version: Annotated[
        int,
        typer.Option("--min-version", help="Regen files below this schema_version"),
    ] = CURRENT_SCHEMA_VERSION,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would regen; don't modify")
    ] = False,
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
    except Exception as e:
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
            text = p.read_text(encoding="utf-8")[:FRONTMATTER_HEAD_BYTES]
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
        text = path.read_text(encoding="utf-8")[:FRONTMATTER_HEAD_BYTES]
    except OSError:
        return None
    match = _FM_URL_RE.search(text)
    return match.group(1) if match else None


_KNOWN_SUBCOMMANDS = frozenset({"run", "regen"})


def _inject_default_subcommand(argv: list[str]) -> list[str]:
    """Insert `run` before the first positional if no subcommand was given.

    Preserves the user-facing `yt2md <URL>` form without needing a default-command
    Click subclass. Top-level flags (--version, --help) and explicit subcommands pass
    through unchanged.
    """
    if not argv:
        return argv
    first = argv[0]
    if first.startswith("-"):  # --version, --help, etc.
        return argv
    if first in _KNOWN_SUBCOMMANDS:
        return argv
    return ["run", *argv]


def main_entry() -> None:
    """Console-script entrypoint: preprocess argv, then hand off to typer."""
    sys.argv = [sys.argv[0], *_inject_default_subcommand(sys.argv[1:])]
    app()
