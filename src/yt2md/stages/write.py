"""Write stage: build deterministic filename + atomically write the markdown."""

from __future__ import annotations

import re
import unicodedata

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
