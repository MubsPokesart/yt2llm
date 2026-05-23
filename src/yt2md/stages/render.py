"""Render a StructuredDoc + Transcript to the final markdown document.

The Jinja2 template owns the document shape; this module owns preprocessing
(YAML-safe filters, paragraph grouping, name substitution, URL building).
"""

from __future__ import annotations

import json
from importlib import resources
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape

if TYPE_CHECKING:
    from yt2md.models import StructuredDoc, Transcript


def _build_env() -> Environment:
    templates_dir = resources.files("yt2md") / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters["yaml_str"] = _yaml_str
    env.filters["yaml_list"] = _yaml_list
    env.filters["mmss"] = _mmss
    env.globals["ytlink"] = _ytlink
    env.globals["emoji_for"] = _emoji_for
    return env


def _yaml_str(value: str) -> str:
    """Emit a YAML-safe double-quoted string."""
    return json.dumps(value, ensure_ascii=False)


def _yaml_list(items: list[str]) -> str:
    """Emit a YAML flow-style list of strings."""
    return "[" + ", ".join(_yaml_str(i) for i in items) + "]"


def _mmss(seconds_value: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS depending on magnitude."""
    total = int(seconds_value)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _ytlink(base_url: str, seconds_value: float) -> str:
    """Build a YouTube deep-link with &t=Ns query suffix."""
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}t={int(seconds_value)}s"


_REFERENCE_EMOJI: dict[str, str] = {
    "book": "📚",
    "paper": "📄",
    "person": "👤",
    "tool": "🛠",
    "video": "🎬",
    "other": "🔗",
}


def _emoji_for(kind: str) -> str:
    """Return the emoji prefix for a Reference.kind value. Falls back to 🔗."""
    return _REFERENCE_EMOJI.get(kind, "🔗")


def render(doc: StructuredDoc, transcript: Transcript) -> str:
    """Build the final markdown document.

    `transcript` is the cleaned transcript used to render the Full Transcript section
    (added in a later task). `doc` provides all analytical sections + frontmatter.
    """
    env = _build_env()
    template = env.get_template("document.md.j2")
    rendered: str = template.render(
        frontmatter=doc.frontmatter,
        tldr=doc.tldr,
        takeaways=doc.takeaways,
        concepts=doc.concepts,
        references=doc.references,
        quotes=doc.quotes,
        sections=doc.sections,
        open_questions=doc.open_questions,
        transcript=transcript,
        speaker_name_map=doc.speaker_name_map,
    )
    return rendered
