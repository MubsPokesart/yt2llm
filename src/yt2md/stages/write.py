"""Write stage: build deterministic filename + atomically write the markdown."""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yt2md.models import Frontmatter

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
