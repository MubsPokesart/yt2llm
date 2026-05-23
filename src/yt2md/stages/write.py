"""Write stage: build deterministic filename + atomically write the markdown."""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from yt2md.models import Frontmatter, StructuredDoc

MAX_SLUG_LENGTH = 80


def slugify(text: str) -> str:
    """Lowercase, ASCII, hyphenated, <=80 chars. Empty input -> empty output.

    Non-ASCII characters are stripped via NFKD normalization. Sequences of
    non-alphanumeric characters collapse to single hyphens. Leading/trailing
    hyphens are trimmed.
    """
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower()
    hyphenated = re.sub(r"[^a-z0-9]+", "-", lowered)
    trimmed = hyphenated.strip("-")
    if len(trimmed) > MAX_SLUG_LENGTH:
        trimmed = trimmed[:MAX_SLUG_LENGTH].rstrip("-")
    return trimmed


def build_filename(fm: Frontmatter) -> str:
    """Build the deterministic output filename for a structured doc.

    Format: {published-date}__{channel-slug}__{title-slug}.md
    On collision (handled by write()), the video_id is appended as a suffix.
    """
    return f"{fm.published.isoformat()}__{slugify(fm.channel)}__{slugify(fm.title)}.md"


def write(*, markdown: str, doc: StructuredDoc, output_dir: Path) -> Path:
    """Write the markdown to the deterministic filename in output_dir.

    Collision: if a file at the target path exists AND its frontmatter `video_id`
    is different, append __{video_id}. If same video_id, overwrite (caller is
    responsible for idempotency short-circuit elsewhere).

    Atomic: writes to a sibling `.tmp` and renames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_filename = build_filename(doc.frontmatter)
    target = output_dir / base_filename

    if target.exists() and _existing_video_id(target) != doc.frontmatter.video_id:
        # Collision with a different video -> append suffix.
        target = output_dir / _add_video_id_suffix(base_filename, doc.frontmatter.video_id)

    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_text(markdown, encoding="utf-8")
        tmp.replace(target)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return target


_FRONTMATTER_VIDEO_ID_RE = re.compile(r"^video_id:\s*(\S+)\s*$", re.MULTILINE)


def _existing_video_id(path: Path) -> str | None:
    """Return the video_id from the frontmatter of an existing file, or None."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _FRONTMATTER_VIDEO_ID_RE.search(text)
    if match is None:
        return None
    return match.group(1)


def _add_video_id_suffix(filename: str, video_id: str) -> str:
    """Insert `__{video_id}` before the `.md` extension."""
    stem, suffix = filename.rsplit(".", 1)
    return f"{stem}__{video_id}.{suffix}"
