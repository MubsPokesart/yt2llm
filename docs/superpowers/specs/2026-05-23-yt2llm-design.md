# yt2llm — Design Spec

**Status:** Design locked; ready for implementation planning.
**Date:** 2026-05-23
**Source:** Derived from `yt2md-spec.md` through deep brainstorming review.

---

## 0. Non-negotiable architectural commitments

These apply to every line of code in this project. They are not preferences. They are not negotiable for "load-bearing" code or anywhere else.

### 0.1 Strict red-green-refactor TDD

- Every behavior change starts with a **failing test, committed first** (either as the first commit of the change, or as the first hunk of a single commit).
- Implementation code is the **minimum** required to flip the failing test green.
- Refactoring happens only with the test suite green.
- **No production code is added without a corresponding test.** This applies to bug fixes, features, refactors, and stage code alike.
- The `superpowers:test-driven-development` skill governs execution.

### 0.2 Strict lint + type enforcement

- `ruff check` with `select = ["ALL"]` and a narrow ignore list (see §11).
- `ruff format --check` enforced.
- `mypy --strict` enforced.
- Tightened Pylint-style thresholds enforced by Ruff:
  - `max-args = 5`
  - `max-statements = 30`
  - `max-branches = 8`
  - `max-complexity = 10` (McCabe)
  - `max-nested-blocks = 4` (preview rule)
  - `max-returns = 6`
  - `max-locals = 15`
- Pre-commit hook: **400 LOC ceiling per Python module** in `src/`. Hitting the ceiling means decompose, not raise.
- **None of these rules are bypassable** with `# noqa`, `# type: ignore`, or local config overrides except for `tests/**` (where assertions and magic numbers are normal) and `src/yt2md/prompts/**` (long lines OK in prompt templates). Every other suppression requires a PR-level justification and is grep-auditable.

### 0.3 Decomposition and intentional architecture

- Every module has one job. If a module's purpose can't be stated in one sentence, decompose it.
- Stages are pure-ish functions with explicit typed inputs and outputs. **No shared mutable state.**
- The pipeline orchestrator is the only code that knows the order of stages. Stages don't know what comes before or after them.
- **No abstraction is added until the third concrete instance exists.** Wrappers, base classes, decorators, dependency-injection helpers, and protocols all require ≥3 callers before they may be introduced.
- **No code is added for hypothetical future use.** Plugin systems, extension points, configurability for unrequested behaviors — all out.
- Public functions are typed; internal helpers prefer concrete types over generics; `typing.cast(...)` is grep-auditable and discouraged.

### 0.4 Why these are firm

The spec calls for an MVP CLI in a single Python codebase. The risk is sprawl — agent-style accretion of one-off branches, scattered conditionals, helper-wrapping-helper indirection — that quickly becomes unmaintainable. The thresholds and TDD discipline above exist to make sprawl mechanically impossible at PR time, not as aspirational ideals.

---

## 1. Goal

Turn any YouTube video URL into a structured markdown file dense enough to:

1. Preserve the **detail** that fades from memory.
2. Live in a Claude **Project** as the corpus for agentic, deep-research-style queries.
3. Surface the specific quote, claim, or reference — and link back to the exact moment in the source video for verification.

MVP signature: **`yt2md <youtube-url>`** — a single URL as the only positional parameter.

---

## 2. Pipeline architecture

### 2.1 Seven stages with on-disk artifact caching

```
download → compress → transcribe → clean → structure → render → write
  (yt-dlp)   (ffmpeg)    (LLM)      (det.)    (LLM)     (det.)   (det.)
```

Each stage is a **plain typed function**. Each produces one or more artifacts written to a deterministic cache path. The cache helper checks for existence before invoking the stage; on hit, the artifact is loaded; on miss, the stage runs and the result is persisted via atomic temp+rename.

```python
# Conceptual shape — actual signatures live in src/yt2md/pipeline.py
def run(url: str, cfg: Config) -> Path:
    video_id = extract_video_id(url)
    paths = ArtifactPaths(cfg.cache_dir, video_id)

    audio_src, metadata = cached(paths.download_pair, lambda: download(url, cfg), ...)
    audio = cached(paths.compressed(cfg), lambda: compress(audio_src, cfg), ...)
    transcript = cached(paths.transcript(audio_hash, cfg), lambda: transcribe(audio, metadata, cfg), ...)
    cleaned = cached(paths.cleaned(transcript_hash, CLEANER_VERSION), lambda: clean(transcript), ...)
    doc = cached(paths.structured(cleaned_hash, PROMPT_VERSION, cfg), lambda: structure(cleaned, metadata, cfg), ...)
    markdown = render(doc, cleaned)
    return write(markdown, doc, cfg)
```

**Properties this design commits to:**

- Pipeline orchestration is **one linear function** readable top to bottom.
- Stages do not know about caching, retries, or each other.
- The cache helper is ~30 LOC. It is **not** a framework.
- Re-running a URL with the same inputs is free (cache hits everywhere).
- Iterating on the structuring prompt only re-runs `structure → render → write`, not transcription.

### 2.2 Cache key strategy

Each artifact path includes a short hash of *everything that affects it*. Stale cache hits are impossible by construction.

- `cache/<video_id>/source_audio.<ext>` — keyed on video_id only (yt-dlp output is what YouTube serves)
- `cache/<video_id>/metadata.json` and `metadata.raw.json` — keyed on video_id
- `cache/<video_id>/audio-<hash(bitrate,codec,channels)>.opus`
- `cache/<video_id>/transcript-<hash(audio_hash,backend,model_id,VOCAB_HINT_VERSION,diarize_flag)>.json` and `.raw.json`
  - `VOCAB_HINT_VERSION` is a module-level integer constant in `src/yt2md/vocab_hint.py`, bumped manually when extraction or formatting logic changes meaning.
- `cache/<video_id>/cleaned-<hash(transcript_hash,CLEANER_VERSION)>.json`
- `cache/<video_id>/structured-<hash(cleaned_hash,PROMPT_VERSION,structuring_model)>.json`
- (no cached `render` artifact — pure, cheap)

`PROMPT_VERSION` and `CLEANER_VERSION` are explicit integer constants in code (`prompts/structure.md` frontmatter and `stages/clean.py` module constant respectively). They are bumped **deliberately** when semantic behavior changes. Whitespace edits to prompts do not bust cache.

`cache/` is **gitignored**. `output/` is **gitignored** unless the user points it elsewhere.

### 2.3 Stage interface — plain functions, not a framework

There is no `Stage` base class, no `@stage` decorator, no plugin registry. Stages have heterogeneous signatures (some return one artifact, some return two; some are LLM-bound, some are pure). A common abstraction would either erase types (lowest common denominator → `Any`) or fan into per-stage variants (one class per stage = wrapper-for-its-own-sake).

The cache helper is the only shared infrastructure stages touch:

```python
# src/yt2md/cache.py — conceptual
def cached(
    path: Path,
    produce: Callable[[], T],
    load: Callable[[Path], T],
    dump: Callable[[T, Path], None],
) -> T:
    if path.exists():
        return load(path)
    result = produce()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    dump(result, tmp)
    tmp.replace(path)  # atomic
    return result
```

---

## 3. Package layout

```
yt2llm/
├── pyproject.toml
├── README.md
├── .pre-commit-config.yaml
├── docs/
│   └── superpowers/specs/...
├── src/yt2md/
│   ├── __init__.py
│   ├── __main__.py              # python -m yt2md entry
│   ├── cli.py                   # typer CLI; thin
│   ├── pipeline.py              # orchestrator; only place that knows stage order
│   ├── models.py                # Pydantic: VideoMetadata, Transcript, StructuredDoc, etc.
│   ├── cache.py                 # cached() helper, artifact path resolution, fingerprinting
│   ├── config.py                # pydantic-settings: env + TOML
│   ├── costs.py                 # one source of truth for per-model rates
│   ├── errors.py                # typed exception hierarchy
│   ├── vocab_hint.py            # transcript prompt construction; two backends
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── download.py          # yt-dlp; raw → normalized metadata; error mapping
│   │   ├── compress.py          # ffmpeg subprocess
│   │   ├── transcribe.py        # backend dispatcher
│   │   ├── transcribe_backends/
│   │   │   ├── openai.py        # gpt-4o-transcribe(-diarize)
│   │   │   └── local.py         # faster-whisper
│   │   ├── chunk.py             # silence-boundary chunking + stitching (used by transcribe when needed)
│   │   ├── clean.py             # pure: Transcript → Transcript; CLEANER_VERSION constant
│   │   ├── structure.py         # Gemini call + Pydantic validation + 1 retry
│   │   ├── render.py            # Jinja2: StructuredDoc + Transcript → markdown string
│   │   └── write.py             # atomic write to output dir
│   ├── prompts/
│   │   ├── structure.md         # version frontmatter: PROMPT_VERSION
│   │   └── vocab_hint.j2        # not needed if format functions live in vocab_hint.py
│   └── templates/
│       └── document.md.j2       # Jinja2 template for final markdown
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── audio/               # short .opus samples
│   │   ├── transcripts/         # normalized + raw JSON
│   │   ├── metadata/            # yt-dlp info_dict samples
│   │   ├── structured/          # StructuredDoc JSON
│   │   └── markdown/            # golden rendered docs
│   ├── unit/
│   ├── integration/
│   └── live/                    # pytest -m live; never in CI
└── .github/workflows/ci.yml
```

**Layout commitments:**

- `pipeline.py` is **the only module that imports from `stages/`** (and from `cache.py`).
- `stages/*.py` modules **never import from each other**.
- `models.py` is a single file owned by no stage. Split only when it crosses ~500 LOC.
- `cli.py` is thin: parse args → build `Config` → call `pipeline.run(url, cfg)`. **`cli.py` does not call stages directly.**
- `prompts/` and `templates/` are text resources, version-controlled separately from code, with version strings tracked in `prompts/structure.md` frontmatter.

---

## 4. Data contracts (Pydantic v2 models)

All Pydantic v2. Validation is the contract; semantic checks live in `models.py` `@field_validator`s where possible, in the consuming stage otherwise.

```python
# src/yt2md/models.py — conceptual

class VideoMetadata(BaseModel):
    video_id: str
    url: str
    title: str
    channel: str
    channel_id: str
    published_date: date            # NOT capture date
    duration_s: float
    description: str
    chapters: list[Chapter]         # may be empty
    tags: list[str]                 # may be empty
    language: str | None            # from yt-dlp; may be None

class Word(BaseModel):
    text: str
    start: float
    end: float
    speaker: str | None             # None if backend doesn't diarize

class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None
    words: list[Word]

class Transcript(BaseModel):
    language: str
    duration_s: float
    backend: Literal["openai_transcribe", "local_whisper"]
    model_id: str                   # e.g., "gpt-4o-transcribe", "faster-whisper-medium"
    chunked: bool                   # True if multi-chunk stitching occurred
    segments: list[Segment]
    speakers: list[str]             # post-clean: collapsed if applicable

class Reference(BaseModel):
    kind: Literal["book", "paper", "person", "tool", "video", "other"]
    name: str
    context: str
    timestamp_s: float

class Takeaway(BaseModel):
    text: str
    timestamp_s: float

class Concept(BaseModel):
    name: str
    definition: str
    timestamp_s: float

class Quote(BaseModel):
    text: str
    speaker: str
    timestamp_s: float

class DetailedSection(BaseModel):
    heading: str
    body: str                       # multi-paragraph
    timestamp_s: float              # section start

class Frontmatter(BaseModel):
    title: str
    channel: str
    url: str
    video_id: str
    published: date
    duration_seconds: int
    captured_at: date
    schema_version: int             # bumped on schema changes; see §10
    genre: Literal["podcast", "lecture", "tutorial", "talk", "interview", "other"]
    speakers: list[str]
    topics: list[str]
    people_mentioned: list[str]
    works_mentioned: list[str]

class StructuredDoc(BaseModel):
    frontmatter: Frontmatter
    tldr: str
    takeaways: list[Takeaway]
    concepts: list[Concept]
    references: list[Reference]
    quotes: list[Quote]
    sections: list[DetailedSection]
    open_questions: list[str]
    speaker_name_map: dict[str, str]    # "SPEAKER_00" → "Andrew Huberman"
```

**Validation rules** (executed in `stages/structure.py` after Gemini returns; retry-once on failure):

- `0.0 ≤ item.timestamp_s ≤ transcript.duration_s` for every timestamped item
- `len(takeaways) ≥ 3`
- `tldr` non-empty
- `frontmatter.title == metadata.title`
- `frontmatter.video_id == metadata.video_id`
- `speakers` (frontmatter) values present in `speaker_name_map.values()` or transcript's raw speaker labels

Validation failure with appended error → 1 retry. Second failure → `InvalidStructuredOutputError`, no cache write.

---

## 5. Stage-by-stage detail

### 5.1 download (yt-dlp)

- Uses `yt-dlp` Python library (not subprocess).
- Produces: `source_audio.<ext>` + `metadata.json` + `metadata.raw.json` (full yt-dlp info_dict).
- Cookie support: `Config.cookies_from_browser: str | None` (e.g., "firefox") and `Config.cookies_file: Path | None` are passed through.
- Error mapping in `_map_ytdlp_error()` (private helper); string-matches yt-dlp messages → typed exceptions (`VideoUnavailableError`, `LivestreamNotEndedError`, etc.). yt-dlp version pinned in `pyproject.toml` to prevent silent breakage.

### 5.2 compress (ffmpeg)

- `subprocess.run(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-c:a", "libopus", "-b:a", "32k", "-application", "voip", out], check=True, capture_output=True, text=True)`.
- No `ffmpeg-python` dependency — adds nothing here.
- Produces: `audio-<hash>.opus`.

### 5.3 transcribe (backend dispatcher)

```python
def transcribe(audio: Path, meta: VideoMetadata, cfg: Config) -> Transcript:
    backend = resolve_backend(cfg)           # respects --backend, auto-fallback
    needs_chunking = _file_exceeds_limits(audio, backend, cfg)
    if needs_chunking:
        chunks = chunk.split_at_silence(audio, backend, cfg)
        transcripts = [_transcribe_one(c, meta, cfg, backend) for c in chunks]
        return chunk.stitch(transcripts, chunks, meta)
    return _transcribe_one(audio, meta, cfg, backend)
```

#### 5.3.1 Backend resolution

```
Config.transcription_backend in {"openai_transcribe", "local_whisper", "auto"}
"auto" → openai if OPENAI_API_KEY present, else local if faster-whisper importable, else ConfigError
CLI flag --backend overrides Config
```

#### 5.3.2 OpenAI backend (`transcribe_backends/openai.py`)

- Model: `gpt-4o-transcribe` (diarize variant).
- `prompt` param: comma-separated glossary via `vocab_hint.format_for_openai()`.
- Tenacity retry: `RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError` → exponential backoff 2s/4s/8s/16s, max 4 attempts.
- Raw response written to `transcript-<hash>.raw.json`; normalized via adapter → `transcript-<hash>.json`.

#### 5.3.3 Local backend (`transcribe_backends/local.py`)

- Library: `faster-whisper`. Default model: `medium`.
- `initial_prompt` param: natural-sentence form via `vocab_hint.format_for_whisper()`.
- No diarization (faster-whisper doesn't support it natively); produces `speakers: []`, all `Word.speaker = None`.
- Optional install: `pip install yt2llm[local]`. If user requests local without it installed, raise `ConfigError("Install with: pip install yt2llm[local]")`.

#### 5.3.4 Chunking (`stages/chunk.py`)

Triggered when `audio_size_mb > 20` OR `duration_s > model_duration_cap`.

- Target chunk duration: `(cap or computed_from_size) × 0.8`.
- Boundaries: search ±30s window around ideal boundary for silence ≥500ms via `ffmpeg -af silencedetect=noise=-30dB:d=0.5`. If none, cut at ideal point.
- Cutting: `ffmpeg -ss <start> -t <duration> -c copy` (no re-encoding).
- Chunks written to `cache/<vid>/chunks/audio_<NN>.opus` + `manifest.json`.
- Transcribe chunks **sequentially** (rate-limit safer; progress reporting linear).
- Stitching: shift all word/segment timestamps by chunk start offset; concatenate.
- Diarization across chunks: **per-chunk labels, not reconciled.** `Transcript.chunked = True` flags the structurer to soften name attribution. Warning emitted to `runs.log`.

### 5.4 clean (deterministic)

```python
# src/yt2md/stages/clean.py
CLEANER_VERSION = 1

HARD_FILLERS = frozenset({"uh", "um", "uhm", "er", "ah"})
# Derived from PodcastFillers (Zhu et al. 2022); covers ~96% of annotated fillers.
# "mm", "mhm", "uh-huh" intentionally excluded — these are backchannel agreement sounds.
# "you know", "like", "I mean" intentionally excluded — high context-dependence,
# removal risks meaning loss in a knowledge-graph artifact.

def clean(transcript: Transcript) -> Transcript:
    """Pure function. Drops filler words, applies 95% speaker collapse."""
```

- Drops words whose `text.lower().strip(punctuation)` ∈ `HARD_FILLERS`.
- Preserves timestamps of surviving words exactly.
- Rebuilds segment text from surviving words; drops empty segments.
- Speaker collapse: if `max(per_speaker_duration) / total_duration ≥ 0.95`, replace all speaker labels with the dominant one. Drop speakers contributing <1% (noise).
- Does **not**: paragraph-group (that's render), fix punctuation (transcriber's job), map names (structurer's job).

### 5.5 structure (Gemini)

- Model: `gemini-3-flash`. `temperature=0.2`. `seed = hash(prompt) % 2**31` (deterministic per prompt). `max_output_tokens=20000`. `safety_settings = BLOCK_NONE` across all categories (we're processing user-supplied source content; default filters block legitimate transcripts).
- `response_schema = StructuredDoc.model_json_schema()` (Gemini-enforced shape).
- Prompt input includes cleaned transcript with inline `[mm:ss]` timestamp markers.
- On `ValidationError` or semantic-validation failure: retry once with the error appended to the prompt. Second failure → `InvalidStructuredOutputError`, no cache write.
- Tenacity retry (same policy as transcribe) for transient SDK errors at the API call boundary.

### 5.6 render (Jinja2)

- `templates/document.md.j2` consumes `StructuredDoc` + `Transcript`.
- Substitutes `speaker_name_map`: every `SPEAKER_NN` in the cleaned transcript becomes the mapped name.
- Paragraph-groups the cleaned transcript into ~60-second blocks with timestamp markers; speaker change forces a new paragraph.
- Builds YouTube `&t=Ns` deep-link URLs from `timestamp_s` values.
- Emoji prefixes for References by `kind`: book = 📚, paper = 📄, person = 👤, tool = 🛠, video = 🎬, other = 🔗.

### 5.7 write (atomic)

- Filename: `{published-date}__{channel-slug}__{title-slug}.md` (slugs: lowercase, ASCII, hyphenated, max 80 chars each).
- On collision (different `video_id`): append `__{video_id}`.
- Idempotency check at `cli.py` entry; this stage is reached only when actually writing.
- Atomic: temp file in same directory + `os.replace` to final path.

---

## 6. Configuration

`pydantic-settings` schema in `src/yt2md/config.py`. Precedence (12-factor): **CLI flag > env var > TOML file > default.**

```python
class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YT2MD_",
        env_file=".env",
        toml_file=["~/.config/yt2md/config.toml", "./yt2md.toml"],  # later wins
    )

    # Secrets
    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr

    # Paths
    output_dir: Path = Path("./output")
    cache_dir: Path = Path("./cache")

    # Audio
    audio_bitrate_kbps: int = 32
    audio_codec: Literal["opus"] = "opus"

    # Transcription
    transcription_backend: Literal["openai_transcribe", "local_whisper", "auto"] = "auto"
    transcription_model: str = "gpt-4o-transcribe"
    local_whisper_model: str = "medium"
    use_transcription_hint: bool = True

    # Structuring
    structuring_model: str = "gemini-3-flash"

    # yt-dlp auth
    cookies_from_browser: str | None = None
    cookies_file: Path | None = None

    # Behavior flags (typically CLI-only)
    force: bool = False
    no_cache: bool = False
```

CLI flags mirror the config: `--backend`, `--cookies-from-browser`, `--cookies`, `--force`, `--no-cache`, `--output-dir`, etc.

---

## 7. Observability

### 7.1 stdout (human, via `rich`)

Live per-stage progress with elapsed time and cost-so-far. Final summary:

```
✓ download           12.3s    8.4 MB audio
✓ compress            0.8s    6.1 MB opus
✓ transcribe         87.2s    $0.362
✓ clean               0.1s
✓ structure          24.7s    $0.041
✓ render              0.0s
✓ write               0.0s    output/2024-03-15__huberman-lab__dopamine.md
Total: 2m 04s, $0.403
```

### 7.2 stderr (machine, via `structlog`)

Structured JSON events. Default level WARNING. `-v` → INFO, `-vv` → DEBUG. Every retry, every cache hit/miss, every stage start/end emits a structured event with `video_id`, `stage`, `attempt` context.

### 7.3 `runs.log` (cost analytics)

One JSONL line per invocation appended to `<cache_dir>/runs.log` on both success and failure:

```json
{"ts":"2026-05-23T14:22:01Z","video_id":"abc123","url":"...","status":"success","duration_s":124.3,"costs":{"transcription_usd":0.362,"transcription_backend":"openai_transcribe","structuring_usd":0.041,"total_usd":0.403},"cache_hits":["audio"],"stages_run":["transcribe","clean","structure","render","write"],"audio_mb":8.4,"duration_video_s":5025,"schema_version":1}
```

Timestamps are UTC ISO-8601. Failure entries include `error_class` and `error_message`. Skip entries (idempotency short-circuit) include `reason: "output_exists"`.

### 7.4 Cost source of truth

`src/yt2md/costs.py` is the single place where per-model rates live. Stages call helpers like `transcription_cost(duration_s, model)` and `gemini_cost(input_tokens, output_tokens, model)`. The orchestrator aggregates; the logger writes.

---

## 8. Error handling

### 8.1 Exception hierarchy (`src/yt2md/errors.py`)

```python
class YT2MDError(Exception): ...

class ConfigError(YT2MDError): ...                    # exit 3

class DownloadError(YT2MDError): ...                  # exit 1
class VideoUnavailableError(DownloadError): ...       # exit 2 (private/removed/age/members/geo)
class LivestreamNotEndedError(DownloadError): ...     # exit 2
class NoAudioStreamError(DownloadError): ...          # exit 2

class TranscriptionError(YT2MDError): ...             # exit 1
class AudioTooLargeError(TranscriptionError): ...     # exit 1

class StructuringError(YT2MDError): ...               # exit 1
class InvalidStructuredOutputError(StructuringError): ...  # exit 1

class WriteError(YT2MDError): ...                     # exit 1
```

### 8.2 Retry policy

- Lives at the SDK call boundary (`transcribe_backends/openai.py` for OpenAI, `stages/structure.py` for Gemini, `stages/download.py` for yt-dlp network errors).
- `tenacity` with exponential backoff (2s/4s/8s/16s, max 4 attempts).
- Retries only: 429, 5xx, timeouts, connection resets.
- **Never** retries: auth failures, validation errors, video-unavailable, content-too-large.
- Every retry logged at WARNING with the underlying error. No silent retries.

### 8.3 CLI catch + exit codes

```python
# src/yt2md/cli.py — conceptual
try:
    path = pipeline.run(url, cfg)
except ConfigError as e:
    typer.echo(f"Configuration error: {e}", err=True); raise typer.Exit(3)
except (VideoUnavailableError, LivestreamNotEndedError, NoAudioStreamError) as e:
    typer.echo(f"{e}", err=True); raise typer.Exit(2)
except YT2MDError as e:
    typer.echo(f"{e.__class__.__name__}: {e}", err=True); raise typer.Exit(1)
```

### 8.4 Mid-pipeline failure recovery

Cache holds completed-upstream artifacts. User fixes the underlying issue (cookies, livestream ended, etc.) and re-runs — pipeline picks up from where it failed. No automatic cleanup.

### 8.5 Mostly-silent video heuristic

After transcribe: `if total_speech_chars / transcript.duration_s < 1.0`: emit WARNING to logger and stdout, add to runs.log, continue. Do not fail.

---

## 9. Idempotency and subcommands

### 9.1 Default behavior

`yt2md <url>` resolves the deterministic output filename. If it exists:
```
Already processed: output/2024-03-15__huberman-lab__dopamine.md
```
Exit 0. Cache untouched.

### 9.2 Overrides

- `--force` — deletes `cache/<video_id>/` (prints confirmation line), runs all stages, overwrites output.
- `--no-cache` — does not read or write cache. Output is written normally.

### 9.3 `yt2md regen` subcommand

- `yt2md regen <path>` — re-runs from `clean` stage onward on the file at `<path>` (assuming its cache subtree exists). Overwrites output. Skips transcription cost.
- `yt2md regen --all [--min-version N]` — scans `output_dir`, regenerates every file whose `schema_version < current_schema_version` (or `< N`).
- `yt2md regen --dry-run` — prints what would regenerate; writes nothing.

---

## 10. Schema versioning

- `Frontmatter.schema_version: int` is written into every output file's frontmatter.
- `CURRENT_SCHEMA_VERSION` is a module-level constant in `models.py`, bumped manually when the schema changes meaning (renamed fields, restructured sections, changed semantics). Adding an optional field does **not** bump.
- Migration runs only via the `regen` subcommand. Never automatic on `yt2md <url>`.

---

## 11. Tooling stack

| Tool | Role |
|---|---|
| `uv` | Project + dependency manager, Python install |
| `ruff check` (with `ALL` + narrow ignores) | Lint |
| `ruff format` | Format |
| `mypy --strict` | Type check |
| `pre-commit` | All of the above on every commit, plus 400-LOC ceiling hook |
| `pytest` + `pytest-cov` + `hypothesis` | Tests + coverage + property tests |
| `tenacity`, `rich`, `structlog`, `Jinja2`, `tiktoken` | Runtime deps (locked in §11.2) |
| `pydantic` v2, `pydantic-settings` | Models + config |
| `typer` | CLI |
| `yt-dlp`, `openai`, `google-genai` | External APIs (pinned versions) |
| Optional: `faster-whisper` | `[local]` extra for offline transcription |

### 11.1 `pyproject.toml` enforcement stanza (canonical)

```toml
[tool.ruff]
line-length = 100
target-version = "py311"
preview = true

[tool.ruff.lint]
select = ["ALL"]
ignore = [
  "D",      # pydocstyle — docstrings not required everywhere
  "COM812", # conflicts with formatter
  "ISC001", # conflicts with formatter
  "FIX",    # allow TODO/FIXME during dev
  "TD",     # ditto
  "ANN101", # self type annotation
  "ANN102", # cls type annotation
]

[tool.ruff.lint.mccabe]
max-complexity = 10

[tool.ruff.lint.pylint]
max-args = 5
max-branches = 8
max-statements = 30
max-returns = 6
max-locals = 15
max-nested-blocks = 4

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101", "PLR2004", "ANN"]
"src/yt2md/prompts/**" = ["E501"]
"src/yt2md/templates/**" = ["E501"]

[tool.mypy]
strict = true
python_version = "3.11"
```

### 11.2 `.pre-commit-config.yaml` (canonical)

```yaml
# Versions (rev: ..., additional_dependencies type stubs) are pinned at implementation time
# to the latest stable releases as of the first commit, then bumped via Renovate or
# manual review. The shape below is canonical; the version pins are not.
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: <pinned-at-impl>
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: <pinned-at-impl>
    hooks:
      - id: mypy
        args: [--strict, src/]
        additional_dependencies:
          - pydantic>=2
          - pydantic-settings
          # plus type stubs identified by `mypy --strict` warnings at impl time
          # (e.g., types-PyYAML if PyYAML is added, etc.)
  - repo: local
    hooks:
      - id: max-file-lines
        name: max file lines (400)
        entry: python -c "import sys; [sys.exit(f'{f}: too long ({sum(1 for _ in open(f))} > 400)') for f in sys.argv[1:] if sum(1 for _ in open(f)) > 400]"
        language: system
        files: ^src/.*\.py$
```

### 11.3 CI (GitHub Actions)

`.github/workflows/ci.yml` runs on push and PR:

```yaml
- uv install
- ruff check --output-format=github
- ruff format --check
- mypy --strict src/
- pytest tests/unit tests/integration --cov=src/yt2md --cov-fail-under=85
```

`tests/live/` is **never run in CI** (no API keys). Runs locally pre-release.

---

## 12. Testing strategy

### 12.1 Discipline

**Strict red-green-refactor TDD per §0.1.** No production code without a corresponding test. The `superpowers:test-driven-development` skill governs execution.

### 12.2 Layers

**Unit tests (`tests/unit/`)** — fast, no network, no large fixtures:

| Module | Coverage |
|---|---|
| `models.py` | Pydantic round-trips, validator semantics |
| `stages/clean.py` | HARD_FILLERS removal preserves surviving timestamps; speaker collapse at 0.949 vs 0.951; empty-segment handling |
| `vocab_hint.py` | Extraction patterns, tiktoken budget enforcement, per-backend formatting |
| `cache.py` | Atomic write semantics, identical-bytes round-trip, parent-dir creation |
| `costs.py` | Per-model rate calculations, aggregation |
| `config.py` | env > TOML > defaults precedence; SecretStr never logged |
| `stages/render.py` | Golden: `render(fixture_doc) == fixture_md`; paragraph grouping at ~60s; speaker substitution; emoji prefix per kind |
| `cli.py` | video_id extraction across URL formats; slug generation |
| `errors.py` | yt-dlp error string → typed exception mapping |

**Integration tests (`tests/integration/`)** — multi-module, no network, SDK boundaries mocked:

- Happy path with mocked stages → golden `.md` output
- Mid-pipeline failure → upstream cached, second run skips upstream
- Idempotency: output exists → skip
- `--force` clears per-video cache
- `--no-cache` writes no cache
- `regen` subcommand re-runs from `clean`
- Chunked transcribe stitching: synthetic chunks → correct global timestamps
- Diarization collapse at threshold boundary
- Edge case error mapping (yt-dlp messages → typed exceptions → correct exit codes)

**Live tests (`tests/live/`)** — `pytest -m live`, requires API keys:

- OpenAI transcribe on 30s fixture
- Gemini structure on small synthetic transcript
- yt-dlp on a known-stable public video

**Property tests (`hypothesis`)** — narrow use:

- `vocab_hint.extract_hints` → token budget never exceeded
- `clean(transcript)` → output timestamps ⊆ input timestamps
- Slug generation → ASCII, ≤80 chars, no leading/trailing hyphens

### 12.3 What we don't test

We do **not** assert on Gemini's specific content choices. The value of structured output is that we test *shape* deterministically and trust the model on *content*.

### 12.4 Fixtures

```
tests/fixtures/
├── audio/
│   ├── short_30s.opus
│   └── multi_speaker_90s.opus
├── transcripts/
│   ├── short_30s.json
│   ├── multi_speaker_90s.json
│   └── chunked_5h_synthetic.json
├── metadata/
│   └── sample_video.json
├── structured/
│   └── sample_doc.json
└── markdown/
    └── sample_doc.md             # golden
```

Audio: 30-second public-domain speech samples (LibriSpeech subset, attributed). Transcripts and structured docs: captured once from real API runs, frozen as golden.

### 12.5 Coverage gate

85% over `src/yt2md/`. SDK-call paths in stage backends excluded via `# pragma: no cover` since they're exercised by live tests.

---

## 13. Out of scope (MVP)

- Web/desktop UI
- Multi-language content (English only)
- Real-time / streaming transcription
- Knowledge-graph backend beyond markdown + RAG (no Neo4j, no Obsidian backlinks)
- Batch / playlist mode (single URL only; shell loop covers technical users)
- Auto-regen on schema change (manual `regen` subcommand only)
- Cross-chunk diarization label reconciliation
- pyannote.audio diarization for the local backend

---

## 14. Open items deferred to implementation

These are implementation craft, not design decisions:

- Exact structuring prompt content (sections, instructions, few-shot examples)
- Exact Jinja2 template for the final markdown
- yt-dlp format selector string (`bestaudio` variants)
- Empirical tuning of chunking thresholds against current API limits (`model_duration_cap`, `max_file_size_mb` safety margin)
- LibriSpeech sample selection for fixtures
- Exact pinned dependency versions in `pyproject.toml` and `.pre-commit-config.yaml`

These are owned by the implementation plan (produced via `superpowers:writing-plans`).

---

## 15. Decision-tree summary (audit trail)

The locked decisions, walked top-down through the design tree:

1. **Pipeline shape:** staged pipeline + on-disk artifact caching (Q1)
2. **Cache key:** hash-based per-stage; gitignored (Q2)
3. **Transcript artifact:** normalized Pydantic + raw OpenAI dump (Q3)
4. **Structurer scope:** analytical-only single Gemini call; separate deterministic cleaner (Q4-Q5)
5. **Package layout:** `(B)` flat infra + `stages/` subfolder (Q6)
6. **Tooling stack:** uv + Ruff `ALL` + mypy strict + pre-commit + tight PLR thresholds + 400 LOC ceiling (Q7)
7. **Stage interface:** plain functions, no framework (Q8)
8. **Error handling:** typed hierarchy + tenacity at SDK boundary (Q9)
9. **Observability:** rich stdout + structlog stderr + JSONL runs.log (Q10)
10. **Foundations:** Pydantic v2, typer, pydantic-settings, Jinja2, tiktoken (Q11)
11. **download/compress split:** separate stages; subprocess for ffmpeg (Q12)
12. **Chunking:** only when needed; silence-boundary; per-chunk diarization labels (Q13)
13. **Local whisper:** faster-whisper, auto-fallback + flag, optional extra (Q14)
14. **Diarization threshold:** 95% duration-weighted, hardcoded (Q15)
15. **Vocab hint:** heuristic extraction; per-backend formats (Q16)
16. **Cleaner:** PodcastFillers-derived HARD_FILLERS only (Q17)
17. **Timestamp grounding:** inline `[mm:ss]` markers (Q18)
18. **Gemini call:** response_schema + Pydantic + 1 retry; temp 0.2; seed from prompt hash; max_output 20000; safety BLOCK_NONE (Q19)
19. **Idempotency:** output-first check; `--force` / `--no-cache` (Q20)
20. **Edge cases:** typed exceptions for each; cookie passthrough (Q21)
21. **Migrations:** schema_version field; manual `regen` subcommand (Q22)
22. **Batch:** out of scope (Q22)
23. **Testing:** unit + integration + live; golden fixtures; hypothesis; **strict red-green TDD** (Q23)

---

**End of design spec.**
