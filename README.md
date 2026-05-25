# yt2llm

Turn YouTube videos into structured, citation-ready markdown for an LLM knowledge graph.

`yt2llm` runs a deterministic, 7-stage pipeline: download a video's audio, compress it,
transcribe with timestamps, scrub fillers, ask a Gemini model to extract analytical
structure (takeaways, concepts, references, quotes, sections, open questions), and render
the result as a single self-contained markdown file with deep-link timestamps.

Every stage's output is cached on disk by a content hash of its inputs, so partial reruns
are free and re-running with the same parameters is a no-op.

## Quick start

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and `ffmpeg` on PATH.

```bash
git clone <repo> && cd yt2llm
uv sync

export OPENAI_API_KEY=sk-...        # for the transcription stage
export GOOGLE_API_KEY=...           # for the structuring stage (required)

uv run yt2md https://www.youtube.com/watch?v=<id>
```

Output lands in `./output/<date>__<channel-slug>__<title-slug>.md`. Re-running the same
URL is a no-op unless you pass `--force`.

For local-whisper fallback (no OpenAI key needed for transcription):

```bash
uv sync --extra local
uv run yt2md run <URL> --backend local_whisper
```

## Usage

```
yt2md <URL>                          # implicit `run` — process a single video
yt2md run <URL> [OPTIONS]            # explicit form
yt2md regen <path-to-output.md>      # regenerate from cached upstream artifacts
yt2md regen --all                    # regenerate every output below CURRENT_SCHEMA_VERSION
```

Common options:

| Option | Effect |
| --- | --- |
| `--cache-dir PATH` | Where stage artifacts live. Default: `./cache` |
| `--output-dir PATH` | Where final `.md` files land. Default: `./output` |
| `--backend NAME` | `openai_transcribe`, `local_whisper`, or `auto` (default) |
| `--cookies-from-browser firefox` | Use browser cookies for age- or members-restricted videos |
| `--cookies PATH` | Use a Netscape-format cookies.txt |
| `--force` | Clear this video's cache and re-run every stage |
| `--no-cache` | Don't read or write the cache for this run |
| `-v` / `-vv` | INFO / DEBUG logging |

## Configuration

Configuration follows a strict precedence: CLI flag &gt; environment variable &gt; default.

API keys accept either the bare conventional name *or* the project-prefixed form (the
prefixed form wins on collision):

| Variable | Accepts | Required |
| --- | --- | --- |
| `OPENAI_API_KEY` or `YT2MD_OPENAI_API_KEY` | OpenAI key | Only if using `openai_transcribe` backend |
| `GOOGLE_API_KEY` or `YT2MD_GOOGLE_API_KEY` | Gemini key | Yes |
| `YT2MD_OUTPUT_DIR` | Path | No (default `./output`) |
| `YT2MD_CACHE_DIR` | Path | No (default `./cache`) |
| `YT2MD_TRANSCRIPTION_MODEL` | OpenAI model id | No (default `whisper-1`) |
| `YT2MD_STRUCTURING_MODEL` | Gemini model id | No (default `gemini-2.5-flash`) |

`whisper-1` is currently the only OpenAI model that returns segment + word timestamps via
`response_format=verbose_json`; the `gpt-4o-transcribe*` family does not, and yt2llm's
markdown shape requires those timestamps.

## How it works

```
URL → download → compress → transcribe → clean → structure → render → write → output.md
        (yt-dlp)   (ffmpeg)   (OpenAI /    (filler-   (Gemini    (Jinja2)   (atomic)
                              faster-      drop +     JSON
                              whisper)     speaker-   schema)
                                           collapse)
```

- **Pipeline orchestration** lives in `src/yt2md/pipeline.py` — the only module that knows
  the stage order.
- **Each stage** in `src/yt2md/stages/` is a pure function over Pydantic v2 models.
  Boundaries are typed contracts (`Transcript`, `StructuredDoc`, ...), not loose dicts.
- **Caching** is content-addressed: each artifact's path includes a hash of every input
  that affects it (`src/yt2md/cache.py`). Change a parameter, get a cache miss.
- **Regeneration** (`yt2md regen`) reads the cached upstream artifacts and re-runs only
  the downstream stages, so you can fix the prompt or the renderer without re-paying for
  transcription.

## Development

```bash
uv sync --all-extras                       # install dev + local extras
uv run pytest                              # full suite (unit + integration)
uv run pre-commit run --all-files          # ruff + ruff-format + mypy + size cap + fast tests
```

The pre-commit chain is mandatory. House rules enforced by tooling:

- `ruff` with `select = ["ALL"]` (with project-specific ignores in `pyproject.toml`)
- `mypy --strict`
- 400-line ceiling per `src/yt2md/` module
- Function caps: 5 args, 30 statements, 8 branches, complexity 10, 4 nested blocks
- Rule of three for abstractions (extract a helper at ≥3 call sites, not before)
- Tests written red-green-refactor; no `--no-verify`

Live tests against real APIs are gated by `pytest -m live` and the bare API-key env vars;
they are skipped by default.

## Status

Functional end-to-end; the surface is stabilising. The current schema_version is `2`
(`yt2md regen --all --min-version 2` will pick up anything older).

The design document and phase plans live in `docs/superpowers/`.
