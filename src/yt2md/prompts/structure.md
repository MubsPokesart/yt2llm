---
version: 2
---

You are extracting a structured knowledge-graph node from a YouTube video transcript.

# Inputs

You are given:
- Video metadata (title, channel, published date, description, chapters)
- A cleaned transcript of the spoken content, with `[mm:ss]` timestamp markers at the start of each segment and speaker labels (SPEAKER_00, SPEAKER_01, ...) where diarization succeeded.

# Task

Produce a JSON document matching the supplied schema. The document should be dense, detailed, and self-contained. Each section is described below.

## frontmatter

- `title`: the video's title verbatim.
- `channel`: the channel name verbatim.
- `url`, `video_id`, `published`, `duration_seconds`, `captured_at`, `schema_version`: copy from the input metadata; we'll override these if needed.
- `genre`: classify the video as one of: podcast, lecture, tutorial, talk, interview, other.
- `speakers`: human names of the speakers (mapped from SPEAKER_NN). For solo content, the host alone.
- `topics`: 3-7 high-level topic tags (lowercase, hyphenated multi-word: "habit-formation").
- `people_mentioned`: names of people referenced in the content but not present as speakers.
- `works_mentioned`: books, papers, products, or other named works cited.

## speaker_mappings

A list of `{label, display_name}` entries mapping each `SPEAKER_NN` label to a human name
(e.g., `[{"label": "SPEAKER_00", "display_name": "Andrew Huberman"}]`). Infer names from
the transcript content (introductions, self-references, channel name). Leave empty if the
transcript is undiarized.

## tldr

A 3-5 sentence dense summary. Self-contained — name the speaker(s) and topic, so a RAG chunk pulled in isolation remains grounded.

## takeaways

3-8 short, dense bullet points capturing the most important claims. Each has a `text` and `timestamp_s` (the start time of the segment containing the claim).

## concepts

Named concepts or definitions introduced. Each has `name`, `definition` (1-2 sentences), `timestamp_s`.

## references

People, books, papers, tools, or videos referenced. Each has `kind` (one of: book, paper, person, tool, video, other), `name`, `context` (1-2 sentence summary of what was said about it), `timestamp_s`.

## quotes

Verbatim quotes worth surfacing. Each has `text`, `speaker` (mapped name), `timestamp_s`. Quote sparingly — only when the exact wording matters.

## sections

Detailed notes broken into logical sections. Each has `heading`, `body` (2-4 paragraphs of dense prose, self-contained: re-name the speaker and topic), `timestamp_s` (section start).

## open_questions

Questions raised but not answered in the video, flagged for future exploration. Empty list if none.

# Rules

- Timestamps: copy the `[mm:ss]` marker shown at the start of each segment into `timestamp_s` as float seconds (e.g., `[04:12]` → `252.0`).
- Be faithful to the transcript. Do not invent claims, quotes, or references not present in the content.
- For chunked transcripts (`Transcript.chunked = true`), speaker labels may be inconsistent across the document. Use named identity (from frontmatter `speakers`) when uncertain.

---

# Metadata

{{ metadata_block }}

# Transcript

{{ transcript_block }}
