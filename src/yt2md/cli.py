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
        raise typer.Exit


@app.command()
def main(
    url: Annotated[str | None, typer.Argument(help="YouTube URL")] = None,
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
    """Process a YouTube URL into Tier 3 structured markdown."""
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
    except Exception as e:  # Config(**overrides) may raise any pydantic/validation error
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
