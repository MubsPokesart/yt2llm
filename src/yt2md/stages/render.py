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
    return env


def _yaml_str(value: str) -> str:
    """Emit a YAML-safe double-quoted string."""
    return json.dumps(value, ensure_ascii=False)


def _yaml_list(items: list[str]) -> str:
    """Emit a YAML flow-style list of strings."""
    return "[" + ", ".join(_yaml_str(i) for i in items) + "]"


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
