"""Structure stage: build Gemini prompt, call Gemini, validate, retry once.

This task (G.1) creates the prompt builder only. G.2 adds semantic validation
and G.3 adds the Gemini SDK call.
"""

from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING

from yt2md.errors import InvalidStructuredOutputError

if TYPE_CHECKING:
    from yt2md.models import StructuredDoc, Transcript, VideoMetadata

PROMPT_VERSION = 1
DESCRIPTION_MAX_CHARS = 1000


def build_structure_prompt(transcript: Transcript, metadata: VideoMetadata) -> str:
    """Render the structuring prompt with metadata and inline-timestamped transcript."""
    template = (resources.files("yt2md") / "prompts" / "structure.md").read_text(encoding="utf-8")
    metadata_block = _format_metadata(metadata)
    transcript_block = _format_transcript(transcript)
    rendered = template.replace("{{ metadata_block }}", metadata_block)
    rendered = rendered.replace("{{ transcript_block }}", transcript_block)
    if transcript.chunked:
        rendered += (
            "\n\nNote: This transcript was produced from multiple chunks. "
            "Speaker labels may be inconsistent across chunks; refer to speakers "
            "by their named identity (from metadata) when in doubt.\n"
        )
    return rendered


def _format_metadata(meta: VideoMetadata) -> str:
    chapters = (
        "\n".join(f"  - {c.title} ({_mmss(c.start_s)}-{_mmss(c.end_s)})" for c in meta.chapters)
        or "  (none)"
    )
    return (
        f"title: {meta.title}\n"
        f"channel: {meta.channel}\n"
        f"published: {meta.published_date}\n"
        f"duration_seconds: {int(meta.duration_s)}\n"
        f"url: {meta.url}\n"
        f"video_id: {meta.video_id}\n"
        f"description: |\n  {meta.description[:DESCRIPTION_MAX_CHARS]}\n"
        f"chapters:\n{chapters}\n"
    )


def _format_transcript(t: Transcript) -> str:
    lines: list[str] = []
    for seg in t.segments:
        prefix = f"[{_mmss(seg.start)}]"
        if seg.speaker:
            lines.append(f"{prefix} {seg.speaker}: {seg.text}")
        else:
            lines.append(f"{prefix} {seg.text}")
    return "\n".join(lines)


def _mmss(seconds_value: float) -> str:
    total = int(seconds_value)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


MIN_TAKEAWAYS = 3


def validate_structured_doc(
    doc: StructuredDoc,
    *,
    transcript: Transcript,
    metadata: VideoMetadata,
) -> None:
    """Semantic validation beyond Pydantic's shape checks. Raises on failure."""
    if len(doc.takeaways) < MIN_TAKEAWAYS:
        msg = f"takeaways: need at least {MIN_TAKEAWAYS}, got {len(doc.takeaways)}"
        raise InvalidStructuredOutputError(msg)

    if not doc.tldr.strip():
        msg = "tldr is empty"
        raise InvalidStructuredOutputError(msg)

    if doc.frontmatter.title != metadata.title:
        msg = f"frontmatter.title {doc.frontmatter.title!r} != metadata.title {metadata.title!r}"
        raise InvalidStructuredOutputError(msg)

    if doc.frontmatter.video_id != metadata.video_id:
        msg = (
            f"frontmatter.video_id {doc.frontmatter.video_id!r} != "
            f"metadata.video_id {metadata.video_id!r}"
        )
        raise InvalidStructuredOutputError(msg)

    duration = transcript.duration_s
    _check_timestamps([t.timestamp_s for t in doc.takeaways], duration, "takeaways")
    _check_timestamps([c.timestamp_s for c in doc.concepts], duration, "concepts")
    _check_timestamps([r.timestamp_s for r in doc.references], duration, "references")
    _check_timestamps([q.timestamp_s for q in doc.quotes], duration, "quotes")
    _check_timestamps([s.timestamp_s for s in doc.sections], duration, "sections")


def _check_timestamps(stamps: list[float], duration_s: float, field_name: str) -> None:
    for ts in stamps:
        if ts < 0.0 or ts > duration_s:
            msg = f"{field_name}: timestamp {ts} out of range [0, {duration_s}]"
            raise InvalidStructuredOutputError(msg)
