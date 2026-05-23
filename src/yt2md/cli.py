"""yt2md CLI — typer entry point. Thin wrapper over pipeline.run()."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Annotated

import click
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


class _DefaultCommandGroup(typer.core.TyperGroup):
    """TyperGroup that routes unknown positional tokens to the callback.

    When the first remaining positional token is not a registered subcommand,
    Click would normally fail with NoSuchCommand.  This subclass detects that
    case and moves ``_protected_args`` back into ``ctx.args`` so the callback
    can pick up the raw URL from ``ctx.args``.
    """

    def invoke(self, ctx: click.Context) -> object:
        protected: list[str] = getattr(ctx, "_protected_args", [])
        if protected and protected[0] not in self.commands:
            ctx.args = protected + list(ctx.args)
            ctx._protected_args = []  # noqa: SLF001
            return click.Command.invoke(self, ctx)
        return super().invoke(ctx)


app = typer.Typer(
    name="yt2md",
    help="Turn YouTube videos into structured markdown for an LLM knowledge graph.",
    add_completion=False,
    no_args_is_help=True,
    cls=_DefaultCommandGroup,
    # allow_extra_args + ignore_unknown_options: unknown tokens pass through to ctx.args.
    # allow_interspersed_args is intentionally left at the default (False) so that options
    # belonging to subcommands (e.g. regen --output-dir) are NOT consumed at the group level.
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
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


def _pop_option(args: list[str], flag: str) -> tuple[str | None, list[str]]:
    """Extract the first ``--flag VALUE`` pair from *args*.

    Returns ``(value, remaining_args)`` where *value* is ``None`` if not found.
    """
    remaining: list[str] = []
    value: str | None = None
    i = 0
    while i < len(args):
        if args[i] == flag and i + 1 < len(args):
            value = args[i + 1]
            i += 2
        else:
            remaining.append(args[i])
            i += 1
    return value, remaining


def _pop_flag(args: list[str], flag: str) -> tuple[bool, list[str]]:
    """Extract a boolean flag from *args*.

    Returns ``(True, remaining_args)`` if *flag* is present, else ``(False, args)``.
    """
    if flag in args:
        return True, [a for a in args if a != flag]
    return False, args


def _absorb_extra_args(
    extra: list[str],
    cache_dir: Path | None,
    output_dir: Path | None,
    backend: str | None,
    cookies_from_browser: str | None,
    cookies_file: Path | None,
    force: bool,
    no_cache: bool,
) -> tuple[str | None, Path | None, Path | None, str | None, str | None, Path | None, bool, bool]:
    """Re-parse options from *extra* tokens (args that fell through after the URL).

    With ``allow_interspersed_args=False``, any option typed after the URL ends up in
    ``ctx.args`` unparsed.  This helper recovers those values and returns updated
    overrides alongside the extracted URL.
    """
    url: str | None = next((a for a in extra if not a.startswith("-")), None)
    if cache_dir is None:
        raw, extra = _pop_option(extra, "--cache-dir")
        if raw is not None:
            cache_dir = Path(raw)
    if output_dir is None:
        raw, extra = _pop_option(extra, "--output-dir")
        if raw is not None:
            output_dir = Path(raw)
    if backend is None:
        raw, extra = _pop_option(extra, "--backend")
        if raw is not None:
            backend = raw
    if cookies_from_browser is None:
        raw, extra = _pop_option(extra, "--cookies-from-browser")
        if raw is not None:
            cookies_from_browser = raw
    if cookies_file is None:
        raw, extra = _pop_option(extra, "--cookies")
        if raw is not None:
            cookies_file = Path(raw)
    if not force:
        found, extra = _pop_flag(extra, "--force")
        if found:
            force = True
    if not no_cache:
        found, extra = _pop_flag(extra, "--no-cache")
        if found:
            no_cache = True
    return url, cache_dir, output_dir, backend, cookies_from_browser, cookies_file, force, no_cache


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
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
    if ctx.invoked_subcommand is not None:
        return

    # ctx.args holds tokens that couldn't be parsed at group level (everything after
    # the URL, since allow_interspersed_args=False stops option parsing at first
    # positional).  Recover the URL and any options that followed it.
    url, cache_dir, output_dir, backend, cookies_from_browser, cookies_file, force, no_cache = (
        _absorb_extra_args(
            list(ctx.args),
            cache_dir,
            output_dir,
            backend,
            cookies_from_browser,
            cookies_file,
            force,
            no_cache,
        )
    )

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

    # Idempotency check
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
