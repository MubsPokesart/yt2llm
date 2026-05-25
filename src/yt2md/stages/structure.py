"""Structure stage: build Gemini prompt, call Gemini, validate, retry once.

This task (G.1) creates the prompt builder only. G.2 adds semantic validation
and G.3 adds the Gemini SDK call.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import TYPE_CHECKING, Any

from google import genai
from google.genai import types as genai_types
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from yt2md.errors import InvalidStructuredOutputError
from yt2md.models import StructuredDoc

if TYPE_CHECKING:
    from yt2md.config import Config
    from yt2md.models import Transcript, VideoMetadata

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


GEMINI_TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 20000
SEED_MODULUS = 2**31
MAX_STRUCTURE_ATTEMPTS = 2


def structure(
    transcript: Transcript,
    metadata: VideoMetadata,
    *,
    cfg: Config,
) -> StructuredDoc:
    """Call Gemini for analytical extraction. Retry once on validation failure.

    Raises InvalidStructuredOutputError if the second attempt also fails.
    """
    prompt = build_structure_prompt(transcript, metadata)
    last_error: Exception | None = None

    for attempt in range(1, MAX_STRUCTURE_ATTEMPTS + 1):
        try:
            response = _call_gemini(prompt, cfg=cfg)
            raw = json.loads(response.text)
            doc = StructuredDoc.model_validate(raw)
            validate_structured_doc(doc, transcript=transcript, metadata=metadata)
        except (ValidationError, InvalidStructuredOutputError, json.JSONDecodeError) as e:
            last_error = e
            if attempt == MAX_STRUCTURE_ATTEMPTS:
                msg = f"Gemini output failed validation after retry: {e}"
                raise InvalidStructuredOutputError(msg) from e
            # Append validation feedback to the prompt and try again.
            prompt = (
                f"{prompt}\n\n# Previous attempt failed validation\n\n{e!s}\n\n"
                "Please retry, fixing the issue above."
            )
            continue
        else:
            return doc

    msg = f"Unreachable: last_error={last_error}"
    raise InvalidStructuredOutputError(msg)


def _call_gemini(prompt: str, *, cfg: Config) -> Any:  # noqa: ANN401
    """Single Gemini API call with tenacity retries on transient SDK errors."""
    return _call_gemini_inner(prompt, cfg)


def _strip_unsupported_schema_keys(node: Any) -> None:  # noqa: ANN401
    """Recursively remove JSON Schema keys that Gemini's Developer API rejects.

    The Gemini SDK validates `response_schema` client-side via
    `_raise_for_unsupported_mldev_properties` and rejects `additionalProperties`
    (only Vertex/Enterprise mode accepts it). Pydantic emits this key for any
    `dict[K, V]` field — e.g., StructuredDoc.speaker_name_map. Strip in place.
    Pydantic-validation of the response (structure.py:154) preserves type safety.
    """
    if isinstance(node, dict):
        node.pop("additionalProperties", None)
        node.pop("additional_properties", None)
        for v in node.values():
            _strip_unsupported_schema_keys(v)
    elif isinstance(node, list):
        for item in node:
            _strip_unsupported_schema_keys(item)


def _build_gemini_schema() -> dict[str, Any]:
    """Build the StructuredDoc JSON schema for Gemini's Developer API."""
    schema: dict[str, Any] = StructuredDoc.model_json_schema()
    _strip_unsupported_schema_keys(schema)
    return schema


@retry(
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _call_gemini_inner(prompt: str, cfg: Config) -> Any:  # noqa: ANN401
    client = genai.Client(api_key=cfg.google_api_key.get_secret_value())
    return client.models.generate_content(
        model=cfg.structuring_model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_build_gemini_schema(),
            temperature=GEMINI_TEMPERATURE,
            seed=hash(prompt) % SEED_MODULUS,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            safety_settings=[
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE",
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE",
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE",
                ),
                genai_types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE",
                ),
            ],
        ),
    )
