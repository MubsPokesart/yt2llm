# yt2llm Phase 2: Deterministic Stages — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Source spec:** `docs/superpowers/specs/2026-05-23-yt2llm-design.md`
**Index:** `docs/superpowers/plans/2026-05-23-yt2llm-index.md`
**Prereq:** Phase 1 complete (errors, models, config, costs, cache modules with passing tests).

**Goal:** Implement the four deterministic stages (`vocab_hint`, `clean`, `render`, `write`) and supporting fixtures, all testable without network access.

**Architecture:** Every stage in this phase is a pure function (or a function that writes to a path you give it). No external APIs. Render uses Jinja2. Write uses atomic temp+rename. Heavy reliance on fixture transcripts and golden markdown for verification.

**Tech Stack:** Adds Jinja2 + tiktoken + hypothesis usage to the Phase 1 stack.

**Definition of done:** All Phase 2 tasks checked off. `lint` + `typecheck` + `cover` all pass. A fixture `StructuredDoc` round-trips to the expected golden markdown byte-for-byte.

---

## Non-negotiable discipline (recap)

Same as Phase 1 — TDD red-green-refactor, lint+typecheck on every commit, 400 LOC ceiling, no abstractions without 3 concrete uses, never `--no-verify`. See `docs/superpowers/plans/2026-05-23-yt2llm-index.md` for the full discipline section.

Shorthand: `lint` = `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`. `typecheck` = `uv run mypy --strict src/`. `test <path>` = `uv run pytest <path> -v`. `cover` = `uv run pytest tests/unit tests/integration --cov=src/yt2md --cov-report=term-missing --cov-fail-under=85`.

---

## Section A: Fixtures

These tasks add reusable test data. They are not behavior — but they're the inputs to most Phase 2 tests, so they come first. Each fixture is small (< 100 lines) and hand-crafted; no captured-from-API data yet (that lands in Phase 3).

---

### Task A.1: Minimal fixture transcript

**Files:**
- Create: `tests/fixtures/transcripts/short_solo.json`
- Create: `tests/conftest.py` (modify — add loader)

A 3-segment, single-speaker, 30-second transcript with one filler word in it.

- [ ] **Step 1: Write the fixture**

`tests/fixtures/transcripts/short_solo.json`:

```json
{
  "language": "en",
  "duration_s": 30.0,
  "backend": "openai_transcribe",
  "model_id": "gpt-4o-transcribe",
  "chunked": false,
  "speakers": ["SPEAKER_00"],
  "segments": [
    {
      "start": 0.0,
      "end": 8.0,
      "text": "Welcome to the show today we discuss dopamine.",
      "speaker": "SPEAKER_00",
      "words": [
        {"text": "Welcome",  "start": 0.0,  "end": 0.6,  "speaker": "SPEAKER_00"},
        {"text": "to",       "start": 0.7,  "end": 0.8,  "speaker": "SPEAKER_00"},
        {"text": "the",      "start": 0.9,  "end": 1.0,  "speaker": "SPEAKER_00"},
        {"text": "show",     "start": 1.1,  "end": 1.5,  "speaker": "SPEAKER_00"},
        {"text": "today",    "start": 1.6,  "end": 2.0,  "speaker": "SPEAKER_00"},
        {"text": "we",       "start": 2.1,  "end": 2.3,  "speaker": "SPEAKER_00"},
        {"text": "discuss",  "start": 2.4,  "end": 3.0,  "speaker": "SPEAKER_00"},
        {"text": "dopamine.","start": 3.1,  "end": 8.0,  "speaker": "SPEAKER_00"}
      ]
    },
    {
      "start": 8.5,
      "end": 18.0,
      "text": "Uh, dopamine signals anticipation, not reward.",
      "speaker": "SPEAKER_00",
      "words": [
        {"text": "Uh,",          "start": 8.5,  "end": 8.7,  "speaker": "SPEAKER_00"},
        {"text": "dopamine",     "start": 9.0,  "end": 9.5,  "speaker": "SPEAKER_00"},
        {"text": "signals",      "start": 9.6,  "end": 10.2, "speaker": "SPEAKER_00"},
        {"text": "anticipation,","start": 10.3, "end": 11.5, "speaker": "SPEAKER_00"},
        {"text": "not",          "start": 11.6, "end": 11.8, "speaker": "SPEAKER_00"},
        {"text": "reward.",      "start": 11.9, "end": 18.0, "speaker": "SPEAKER_00"}
      ]
    },
    {
      "start": 18.5,
      "end": 30.0,
      "text": "It peaks before the reward arrives, not during.",
      "speaker": "SPEAKER_00",
      "words": [
        {"text": "It",      "start": 18.5, "end": 18.7, "speaker": "SPEAKER_00"},
        {"text": "peaks",   "start": 18.8, "end": 19.2, "speaker": "SPEAKER_00"},
        {"text": "before",  "start": 19.3, "end": 19.8, "speaker": "SPEAKER_00"},
        {"text": "the",     "start": 19.9, "end": 20.0, "speaker": "SPEAKER_00"},
        {"text": "reward",  "start": 20.1, "end": 20.6, "speaker": "SPEAKER_00"},
        {"text": "arrives,","start": 20.7, "end": 21.5, "speaker": "SPEAKER_00"},
        {"text": "not",     "start": 21.6, "end": 21.8, "speaker": "SPEAKER_00"},
        {"text": "during.", "start": 21.9, "end": 30.0, "speaker": "SPEAKER_00"}
      ]
    }
  ]
}
```

- [ ] **Step 2: Add a loader fixture to `tests/conftest.py`**

Append to the existing `conftest.py`:

```python
import json

from yt2md.models import Transcript, VideoMetadata


@pytest.fixture
def short_solo_transcript() -> Transcript:
    """3-segment solo transcript with one filler ('Uh,'). Loaded from JSON fixture."""
    data = json.loads((FIXTURES_DIR / "transcripts" / "short_solo.json").read_text(encoding="utf-8"))
    return Transcript.model_validate(data)
```

- [ ] **Step 3: Add a smoke test that the fixture loads**

`tests/unit/test_fixture_loading.py`:

```python
"""Smoke tests: fixtures load into Pydantic models without error."""

from yt2md.models import Transcript


def test_short_solo_transcript_loads(short_solo_transcript: Transcript) -> None:
    assert short_solo_transcript.duration_s == 30.0
    assert short_solo_transcript.speakers == ["SPEAKER_00"]
    assert len(short_solo_transcript.segments) == 3
    # Segment 2 starts with the filler "Uh,"
    assert short_solo_transcript.segments[1].words[0].text == "Uh,"
```

- [ ] **Step 4: Run + lint + commit**

```bash
uv run pytest tests/unit/test_fixture_loading.py -v
uv run ruff check src/ tests/
uv run mypy --strict src/
git add tests/fixtures/transcripts/short_solo.json tests/conftest.py tests/unit/test_fixture_loading.py
git commit -m "$(cat <<'EOF'
test(fixtures): add short solo transcript fixture and loader

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task A.2: Multi-speaker fixture transcript

**Files:**
- Create: `tests/fixtures/transcripts/multi_speaker.json`
- Modify: `tests/conftest.py` (add loader)

Two-speaker transcript with realistic per-speaker duration distribution: 96% one speaker, 4% the other (to test the 95% collapse threshold boundary).

- [ ] **Step 1: Write the fixture**

`tests/fixtures/transcripts/multi_speaker.json`:

```json
{
  "language": "en",
  "duration_s": 100.0,
  "backend": "openai_transcribe",
  "model_id": "gpt-4o-transcribe",
  "chunked": false,
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "segments": [
    {
      "start": 0.0,
      "end": 96.0,
      "text": "Long monologue by speaker zero spanning ninety six seconds total.",
      "speaker": "SPEAKER_00",
      "words": [
        {"text": "Long",      "start": 0.0,  "end": 12.0, "speaker": "SPEAKER_00"},
        {"text": "monologue", "start": 12.0, "end": 24.0, "speaker": "SPEAKER_00"},
        {"text": "by",        "start": 24.0, "end": 36.0, "speaker": "SPEAKER_00"},
        {"text": "speaker",   "start": 36.0, "end": 48.0, "speaker": "SPEAKER_00"},
        {"text": "zero.",     "start": 48.0, "end": 96.0, "speaker": "SPEAKER_00"}
      ]
    },
    {
      "start": 96.0,
      "end": 100.0,
      "text": "Brief reply.",
      "speaker": "SPEAKER_01",
      "words": [
        {"text": "Brief",  "start": 96.0, "end": 98.0,  "speaker": "SPEAKER_01"},
        {"text": "reply.", "start": 98.0, "end": 100.0, "speaker": "SPEAKER_01"}
      ]
    }
  ]
}
```

Per-speaker duration: SPEAKER_00 = 96s, SPEAKER_01 = 4s. SPEAKER_00 dominance = 96/100 = 0.96 → above the 95% threshold → should collapse.

- [ ] **Step 2: Add loader and test**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def multi_speaker_transcript() -> Transcript:
    """2-speaker transcript with 96/4% duration split (above 95% collapse threshold)."""
    data = json.loads((FIXTURES_DIR / "transcripts" / "multi_speaker.json").read_text(encoding="utf-8"))
    return Transcript.model_validate(data)
```

Append to `tests/unit/test_fixture_loading.py`:

```python
def test_multi_speaker_transcript_loads(multi_speaker_transcript: Transcript) -> None:
    assert multi_speaker_transcript.duration_s == 100.0
    assert multi_speaker_transcript.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert len(multi_speaker_transcript.segments) == 2
```

- [ ] **Step 3: Run + lint + commit**

```bash
uv run pytest tests/unit/test_fixture_loading.py -v
uv run ruff check src/ tests/
uv run mypy --strict src/
git add tests/fixtures/transcripts/multi_speaker.json tests/conftest.py tests/unit/test_fixture_loading.py
git commit -m "$(cat <<'EOF'
test(fixtures): add multi-speaker transcript at 96/4% duration split

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task A.3: Sample VideoMetadata fixture

**Files:**
- Create: `tests/fixtures/metadata/huberman_sample.json`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write fixture**

`tests/fixtures/metadata/huberman_sample.json`:

```json
{
  "video_id": "abc123",
  "url": "https://www.youtube.com/watch?v=abc123",
  "title": "Dopamine, Motivation & Drive",
  "channel": "Huberman Lab",
  "channel_id": "UCxxxx",
  "published_date": "2024-03-15",
  "duration_s": 5025.0,
  "description": "In this Huberman Lab episode, Andrew Huberman discusses dopamine, motivation and drive. References The Molecule of More by Daniel Lieberman, and the work of Robert Sapolsky at Stanford.",
  "chapters": [
    {"title": "Introduction", "start_s": 0.0, "end_s": 60.0},
    {"title": "What dopamine actually does", "start_s": 60.0, "end_s": 2700.0},
    {"title": "Tools to raise baseline dopamine", "start_s": 2700.0, "end_s": 5025.0}
  ],
  "tags": ["neuroscience", "dopamine", "motivation"],
  "language": "en"
}
```

- [ ] **Step 2: Add loader**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def huberman_metadata() -> VideoMetadata:
    """Sample VideoMetadata mimicking a Huberman Lab episode."""
    data = json.loads((FIXTURES_DIR / "metadata" / "huberman_sample.json").read_text(encoding="utf-8"))
    return VideoMetadata.model_validate(data)
```

Add test:

```python
def test_huberman_metadata_loads(huberman_metadata: VideoMetadata) -> None:
    assert huberman_metadata.title == "Dopamine, Motivation & Drive"
    assert huberman_metadata.channel == "Huberman Lab"
    assert len(huberman_metadata.chapters) == 3
```

- [ ] **Step 3: Run + lint + commit**

```bash
uv run pytest tests/unit/test_fixture_loading.py -v
uv run ruff check src/ tests/
uv run mypy --strict src/
git add tests/fixtures/metadata/huberman_sample.json tests/conftest.py tests/unit/test_fixture_loading.py
git commit -m "$(cat <<'EOF'
test(fixtures): add Huberman Lab sample VideoMetadata fixture

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section B: vocab_hint.py

---

### Task B.1: Module skeleton + VocabularyHints dataclass + VOCAB_HINT_VERSION

**Files:**
- Create: `tests/unit/test_vocab_hint_model.py`
- Create: `src/yt2md/vocab_hint.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_vocab_hint_model.py`:

```python
"""Tests for the VocabularyHints dataclass and VOCAB_HINT_VERSION constant."""

from yt2md.vocab_hint import VOCAB_HINT_VERSION, VocabularyHints


class TestVocabHintVersion:
    def test_is_positive_int(self) -> None:
        assert isinstance(VOCAB_HINT_VERSION, int)
        assert VOCAB_HINT_VERSION >= 1


class TestVocabularyHints:
    def test_empty(self) -> None:
        h = VocabularyHints(
            people=[],
            works=[],
            concepts=[],
            organizations=[],
            channel="",
            title="",
        )
        assert h.people == []

    def test_populated(self) -> None:
        h = VocabularyHints(
            people=["Andrew Huberman"],
            works=["The Molecule of More"],
            concepts=["dopamine"],
            organizations=["Stanford"],
            channel="Huberman Lab",
            title="Dopamine",
        )
        assert h.people == ["Andrew Huberman"]
        assert h.channel == "Huberman Lab"
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_vocab_hint_model.py -v
```

Expected: FAIL — module does not exist.

- [ ] **Step 3: Write `src/yt2md/vocab_hint.py`**

```python
"""Vocabulary hint construction for transcription backends.

The OpenAI gpt-4o-transcribe `prompt` param and faster-whisper `initial_prompt` param
accept ~224 tokens of vocabulary/style biasing. Empirically validated by PodcastFillers
and OpenAI cookbook guidance:
  - OpenAI: comma-separated glossary, ~"Important words: A, B, C."
  - Whisper: natural sentence form ("This is a transcript of ... discussing ...")
    so the model mimics style and capitalization.

VOCAB_HINT_VERSION is bumped manually when extraction logic or per-backend formatting
changes meaning. Bumping invalidates the transcript cache key.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VOCAB_HINT_VERSION = 1


@dataclass(frozen=True)
class VocabularyHints:
    """Structured vocabulary extracted from video metadata.

    Categorization powers per-backend formatting:
      - people: subject/object in Whisper sentences; first in OpenAI glossary
      - works: italicized in Whisper sentences; second in OpenAI glossary
      - organizations: locations/affiliations
      - concepts: topics, technical terms, acronyms
      - channel + title: always included as opening context
    """

    people: list[str] = field(default_factory=list)
    works: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    channel: str = ""
    title: str = ""
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_vocab_hint_model.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/vocab_hint.py tests/unit/test_vocab_hint_model.py
git commit -m "$(cat <<'EOF'
feat(vocab_hint): add VocabularyHints dataclass and VOCAB_HINT_VERSION

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B.2: extract_hints — Title Case people / proper nouns

**Files:**
- Create: `tests/unit/test_vocab_hint_extract.py`
- Modify: `src/yt2md/vocab_hint.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_vocab_hint_extract.py`:

```python
"""Tests for vocab_hint.extract_hints — categorized extraction from VideoMetadata."""

from datetime import date

from yt2md.models import VideoMetadata
from yt2md.vocab_hint import extract_hints


def _meta(*, title: str = "T", channel: str = "C", description: str = "") -> VideoMetadata:
    return VideoMetadata(
        video_id="v",
        url="https://www.youtube.com/watch?v=v",
        title=title,
        channel=channel,
        channel_id="UC1",
        published_date=date(2025, 1, 1),
        duration_s=60.0,
        description=description,
        chapters=[],
        tags=[],
        language="en",
    )


class TestExtractPeople:
    def test_title_case_two_words(self) -> None:
        m = _meta(description="Featuring Andrew Huberman from Stanford.")
        h = extract_hints(m)
        assert "Andrew Huberman" in h.people

    def test_title_case_three_words(self) -> None:
        m = _meta(description="An interview with Mary Lou Jepsen.")
        h = extract_hints(m)
        assert "Mary Lou Jepsen" in h.people

    def test_lowercase_not_extracted(self) -> None:
        m = _meta(description="just regular sentence content here")
        h = extract_hints(m)
        assert h.people == []


class TestExtractTitleAndChannel:
    def test_title_present(self) -> None:
        m = _meta(title="Dopamine and Drive")
        h = extract_hints(m)
        assert h.title == "Dopamine and Drive"

    def test_channel_present(self) -> None:
        m = _meta(channel="Huberman Lab")
        h = extract_hints(m)
        assert h.channel == "Huberman Lab"
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_vocab_hint_extract.py -v
```

Expected: FAIL — `extract_hints` does not exist.

- [ ] **Step 3: Append to `src/yt2md/vocab_hint.py`**

Add at the top imports:

```python
import re
```

Append at the bottom:

```python
# Title Case sequence: capitalized word followed by 1-3 more capitalized words.
# Tightened to require ≥2 words so single-word capitalized common nouns
# ("Welcome") don't pollute the people list.
_TITLE_CASE_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")


def extract_hints(meta: VideoMetadata) -> VocabularyHints:
    """Extract a categorized VocabularyHints from video metadata.

    Sources (priority order): title > channel > chapter titles > first 500 chars
    of description. URLs are stripped before scanning.
    """
    desc_excerpt = _strip_urls(meta.description)[:500]
    sources = [meta.title, *(c.title for c in meta.chapters), desc_excerpt]

    people = _dedup_ordered(
        match
        for src in sources
        for match in _TITLE_CASE_PATTERN.findall(src)
    )

    return VocabularyHints(
        people=people,
        works=[],
        concepts=[],
        organizations=[],
        channel=meta.channel,
        title=meta.title,
    )


def _strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+", "", text)


def _dedup_ordered(items: object) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:  # type: ignore[union-attr]
        s = str(item)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
```

Add `from yt2md.models import VideoMetadata` near the top imports.

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_vocab_hint_extract.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/vocab_hint.py tests/unit/test_vocab_hint_extract.py
git commit -m "$(cat <<'EOF'
feat(vocab_hint): extract Title Case people from metadata sources

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B.3: extract_hints — quoted works + acronyms + CamelCase concepts

**Files:**
- Modify: `tests/unit/test_vocab_hint_extract.py` (append)
- Modify: `src/yt2md/vocab_hint.py` (extend `extract_hints`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_vocab_hint_extract.py`:

```python
class TestExtractWorks:
    def test_double_quoted_extracted_as_work(self) -> None:
        m = _meta(description='Discussing "The Molecule of More" by Daniel Lieberman.')
        h = extract_hints(m)
        assert "The Molecule of More" in h.works

    def test_smart_quotes_extracted(self) -> None:
        m = _meta(description="Discussing “The Molecule of More” at length.")
        h = extract_hints(m)
        assert "The Molecule of More" in h.works


class TestExtractAcronyms:
    def test_short_acronyms_extracted(self) -> None:
        m = _meta(description="The fMRI scans showed ADHD signatures and DNA damage.")
        h = extract_hints(m)
        # 2-5 char all-caps tokens
        assert "ADHD" in h.concepts
        assert "DNA" in h.concepts

    def test_single_letter_not_extracted(self) -> None:
        m = _meta(description="A study of A and B groups.")
        h = extract_hints(m)
        assert "A" not in h.concepts
        assert "B" not in h.concepts

    def test_six_letter_not_extracted(self) -> None:
        m = _meta(description="SHOULDNT extract this.")
        h = extract_hints(m)
        assert "SHOULDNT" not in h.concepts


class TestExtractCamelCase:
    def test_camel_case_extracted_as_concept(self) -> None:
        m = _meta(description="Using PyTorch and TensorFlow for the experiment.")
        h = extract_hints(m)
        assert "PyTorch" in h.concepts
        assert "TensorFlow" in h.concepts

    def test_alphanumeric_with_internal_digit(self) -> None:
        m = _meta(description="GPT-4 and Claude-3 are competing models.")
        h = extract_hints(m)
        assert "GPT-4" in h.concepts


class TestDedup:
    def test_repeated_term_deduped(self) -> None:
        m = _meta(description="Andrew Huberman explained. Andrew Huberman emphasized.")
        h = extract_hints(m)
        assert h.people.count("Andrew Huberman") == 1
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_vocab_hint_extract.py -v
```

Expected: new tests FAIL.

- [ ] **Step 3: Extend `src/yt2md/vocab_hint.py`**

Add additional patterns above `extract_hints`:

```python
# Quoted phrases (single, double, or smart quotes) — likely titles of works.
_QUOTED_PATTERN = re.compile(
    r'["“]([^"”]{2,80})["”]'
)

# All-caps acronyms, 2-5 chars. Single letters and longer strings excluded
# (longer all-caps is usually shouting or noise; single letters are stop-words).
_ACRONYM_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")

# CamelCase: starts with capital, has at least one lowercase, then at least one capital
# OR alphanumeric with internal digit/hyphen ("GPT-4", "Claude-3").
_CAMELCASE_PATTERN = re.compile(
    r"\b([A-Z][a-z]+[A-Z][A-Za-z0-9]*|[A-Z][A-Za-z]+[-]?\d+)\b"
)
```

Replace the body of `extract_hints` with:

```python
def extract_hints(meta: VideoMetadata) -> VocabularyHints:
    """Extract a categorized VocabularyHints from video metadata.

    Sources (priority order): title > channel > chapter titles > first 500 chars
    of description. URLs are stripped before scanning.
    """
    desc_excerpt = _strip_urls(meta.description)[:500]
    sources = [meta.title, *(c.title for c in meta.chapters), desc_excerpt]
    combined = "\n".join(sources)

    works = _dedup_ordered(_QUOTED_PATTERN.findall(combined))
    acronyms = _dedup_ordered(_ACRONYM_PATTERN.findall(combined))
    camel = _dedup_ordered(_CAMELCASE_PATTERN.findall(combined))
    concepts = _dedup_ordered([*acronyms, *camel])
    people = _dedup_ordered(_TITLE_CASE_PATTERN.findall(combined))

    return VocabularyHints(
        people=people,
        works=works,
        concepts=concepts,
        organizations=[],
        channel=meta.channel,
        title=meta.title,
    )
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_vocab_hint_extract.py -v
```

Expected: all tests PASS. Note: `Mary Lou Jepsen` test from B.2 still passes because Title Case pattern still matches.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/vocab_hint.py tests/unit/test_vocab_hint_extract.py
git commit -m "$(cat <<'EOF'
feat(vocab_hint): extract works (quoted), acronyms, CamelCase concepts

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B.4: format_for_openai (comma-separated glossary)

**Files:**
- Create: `tests/unit/test_vocab_hint_format_openai.py`
- Modify: `src/yt2md/vocab_hint.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_vocab_hint_format_openai.py`:

```python
"""Tests for vocab_hint.format_for_openai — comma-separated glossary."""

from yt2md.vocab_hint import VocabularyHints, format_for_openai


def _hints(**kwargs: object) -> VocabularyHints:
    defaults: dict[str, object] = {
        "people": [],
        "works": [],
        "concepts": [],
        "organizations": [],
        "channel": "C",
        "title": "T",
    }
    defaults.update(kwargs)
    return VocabularyHints(**defaults)  # type: ignore[arg-type]


class TestFormatForOpenAI:
    def test_starts_with_glossary_framing(self) -> None:
        h = _hints(people=["Andrew Huberman"])
        out = format_for_openai(h)
        assert out.lower().startswith("glossary")

    def test_includes_people(self) -> None:
        h = _hints(people=["Andrew Huberman", "Robert Sapolsky"])
        out = format_for_openai(h)
        assert "Andrew Huberman" in out
        assert "Robert Sapolsky" in out

    def test_includes_works(self) -> None:
        h = _hints(works=["The Molecule of More"])
        out = format_for_openai(h)
        assert "The Molecule of More" in out

    def test_includes_concepts(self) -> None:
        h = _hints(concepts=["dopamine", "GPT-4"])
        out = format_for_openai(h)
        assert "dopamine" in out
        assert "GPT-4" in out

    def test_includes_title(self) -> None:
        h = _hints(title="Dopamine and Drive")
        out = format_for_openai(h)
        assert "Dopamine and Drive" in out

    def test_includes_channel(self) -> None:
        h = _hints(channel="Huberman Lab")
        out = format_for_openai(h)
        assert "Huberman Lab" in out

    def test_comma_separated(self) -> None:
        h = _hints(people=["A", "B"], concepts=["C"])
        out = format_for_openai(h)
        # No newlines; commas everywhere
        assert "\n" not in out
        assert "," in out

    def test_empty_hints_still_returns_string(self) -> None:
        h = _hints()
        out = format_for_openai(h)
        assert isinstance(out, str)
        assert "T" in out  # title
        assert "C" in out  # channel
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_vocab_hint_format_openai.py -v
```

Expected: FAIL — `format_for_openai` does not exist.

- [ ] **Step 3: Append to `src/yt2md/vocab_hint.py`**

```python
DEFAULT_TOKEN_BUDGET = 220


def format_for_openai(hints: VocabularyHints, *, max_tokens: int = DEFAULT_TOKEN_BUDGET) -> str:
    """Format hints as a comma-separated glossary for gpt-4o-transcribe's `prompt` param.

    Per OpenAI guidance: short keyword lists work better than instructions for the
    transcription `prompt` parameter. Truncates to `max_tokens` via tiktoken.
    """
    parts: list[str] = []
    if hints.title:
        parts.append(hints.title)
    if hints.channel:
        parts.append(hints.channel)
    parts.extend(hints.people)
    parts.extend(hints.works)
    parts.extend(hints.organizations)
    parts.extend(hints.concepts)
    body = ", ".join(p for p in parts if p)
    full = f"Glossary for transcription: {body}"
    return _truncate_to_tokens(full, max_tokens)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate `text` so the token count does not exceed `max_tokens`.

    Uses tiktoken's cl100k_base encoding (OpenAI's standard).
    """
    import tiktoken  # local import: avoid module load cost when not needed

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_vocab_hint_format_openai.py -v
```

Expected: all 8 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/vocab_hint.py tests/unit/test_vocab_hint_format_openai.py
git commit -m "$(cat <<'EOF'
feat(vocab_hint): format_for_openai produces tiktoken-truncated glossary

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B.5: format_for_whisper (natural-sentence style)

**Files:**
- Create: `tests/unit/test_vocab_hint_format_whisper.py`
- Modify: `src/yt2md/vocab_hint.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_vocab_hint_format_whisper.py`:

```python
"""Tests for vocab_hint.format_for_whisper — natural-sentence style."""

from yt2md.vocab_hint import VocabularyHints, format_for_whisper


def _hints(**kwargs: object) -> VocabularyHints:
    defaults: dict[str, object] = {
        "people": [],
        "works": [],
        "concepts": [],
        "organizations": [],
        "channel": "C",
        "title": "T",
    }
    defaults.update(kwargs)
    return VocabularyHints(**defaults)  # type: ignore[arg-type]


class TestFormatForWhisper:
    def test_starts_with_transcript_framing(self) -> None:
        h = _hints(people=["Andrew Huberman"], channel="Huberman Lab")
        out = format_for_whisper(h)
        # Whisper mimics style. Frame as a transcript description.
        assert "transcript" in out.lower()

    def test_includes_channel_in_sentence(self) -> None:
        h = _hints(channel="Huberman Lab")
        out = format_for_whisper(h)
        assert "Huberman Lab" in out

    def test_includes_people_in_sentence(self) -> None:
        h = _hints(people=["Andrew Huberman"], channel="Huberman Lab")
        out = format_for_whisper(h)
        assert "Andrew Huberman" in out

    def test_includes_works(self) -> None:
        h = _hints(works=["The Molecule of More"])
        out = format_for_whisper(h)
        assert "The Molecule of More" in out

    def test_preserves_capitalization(self) -> None:
        h = _hints(concepts=["GPT-4", "PyTorch"])
        out = format_for_whisper(h)
        # Whisper picks up capitalization from prompt; must preserve exactly.
        assert "GPT-4" in out
        assert "PyTorch" in out
        assert "gpt-4" not in out
        assert "pytorch" not in out

    def test_ends_with_period(self) -> None:
        h = _hints(people=["A"], channel="C", title="T")
        out = format_for_whisper(h).strip()
        assert out.endswith(".")

    def test_empty_hints_returns_string(self) -> None:
        h = _hints()
        out = format_for_whisper(h)
        assert isinstance(out, str)
        assert out.strip()  # non-empty
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_vocab_hint_format_whisper.py -v
```

Expected: FAIL.

- [ ] **Step 3: Append to `src/yt2md/vocab_hint.py`**

```python
def format_for_whisper(hints: VocabularyHints, *, max_tokens: int = DEFAULT_TOKEN_BUDGET) -> str:
    """Format hints as a natural-language paragraph for Whisper's `initial_prompt`.

    Whisper mimics style/capitalization rather than following instructions.
    Sentences are constructed so the names appear in natural grammatical contexts.
    """
    sentences: list[str] = []

    opener = _build_opener(hints)
    if opener:
        sentences.append(opener)

    if hints.people:
        speakers_clause = (
            f"The speakers include {_join_oxford(hints.people)}."
        )
        sentences.append(speakers_clause)

    if hints.works:
        works_clause = (
            f"Works referenced include {_join_oxford(hints.works)}."
        )
        sentences.append(works_clause)

    if hints.organizations:
        orgs_clause = (
            f"Affiliations mentioned: {_join_oxford(hints.organizations)}."
        )
        sentences.append(orgs_clause)

    if not sentences:
        # Fallback minimal sentence — Whisper expects SOME content.
        sentences.append(f"This is a transcript from {hints.channel or 'a video'}.")

    full = " ".join(sentences)
    return _truncate_to_tokens(full, max_tokens)


def _build_opener(hints: VocabularyHints) -> str:
    if hints.channel and hints.title:
        return (
            f"This is a transcript of an episode of {hints.channel} titled "
            f'"{hints.title}".'
        )
    if hints.channel:
        return f"This is a transcript from {hints.channel}."
    if hints.title:
        return f'This is a transcript of "{hints.title}".'
    return ""


def _join_oxford(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_vocab_hint_format_whisper.py -v
```

Expected: 7 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/vocab_hint.py tests/unit/test_vocab_hint_format_whisper.py
git commit -m "$(cat <<'EOF'
feat(vocab_hint): format_for_whisper produces natural-sentence style

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task B.6: Property test — token budget never exceeded

**Files:**
- Create: `tests/unit/test_vocab_hint_property.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_vocab_hint_property.py`:

```python
"""Property tests: token budget is always respected regardless of input."""

import tiktoken
from hypothesis import given
from hypothesis import strategies as st

from yt2md.vocab_hint import VocabularyHints, format_for_openai, format_for_whisper


_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


# Reasonable strategy: short strings, bounded lists.
_short_str = st.text(min_size=0, max_size=40)
_str_list = st.lists(_short_str, min_size=0, max_size=20)


@given(
    people=_str_list,
    works=_str_list,
    concepts=_str_list,
    organizations=_str_list,
    channel=_short_str,
    title=_short_str,
)
def test_openai_format_respects_budget(
    people: list[str],
    works: list[str],
    concepts: list[str],
    organizations: list[str],
    channel: str,
    title: str,
) -> None:
    h = VocabularyHints(
        people=people,
        works=works,
        concepts=concepts,
        organizations=organizations,
        channel=channel,
        title=title,
    )
    out = format_for_openai(h, max_tokens=50)
    assert _count_tokens(out) <= 50


@given(
    people=_str_list,
    works=_str_list,
    concepts=_str_list,
    organizations=_str_list,
    channel=_short_str,
    title=_short_str,
)
def test_whisper_format_respects_budget(
    people: list[str],
    works: list[str],
    concepts: list[str],
    organizations: list[str],
    channel: str,
    title: str,
) -> None:
    h = VocabularyHints(
        people=people,
        works=works,
        concepts=concepts,
        organizations=organizations,
        channel=channel,
        title=title,
    )
    out = format_for_whisper(h, max_tokens=50)
    assert _count_tokens(out) <= 50
```

- [ ] **Step 2: Run — confirm passes**

```bash
uv run pytest tests/unit/test_vocab_hint_property.py -v
```

Expected: 2 PASS (both properties hold). If FAIL, the truncation logic in `_truncate_to_tokens` has a bug.

- [ ] **Step 3: Lint + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add tests/unit/test_vocab_hint_property.py
git commit -m "$(cat <<'EOF'
test(vocab_hint): property test that token budget is never exceeded

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section C: clean.py

---

### Task C.1: Constants + clean() skeleton (no-op)

**Files:**
- Create: `tests/unit/test_clean_skeleton.py`
- Create: `src/yt2md/stages/__init__.py`
- Create: `src/yt2md/stages/clean.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_clean_skeleton.py`:

```python
"""Tests for the clean stage skeleton and module constants."""

from yt2md.models import Transcript
from yt2md.stages.clean import CLEANER_VERSION, HARD_FILLERS, clean


class TestConstants:
    def test_cleaner_version_is_positive_int(self) -> None:
        assert isinstance(CLEANER_VERSION, int)
        assert CLEANER_VERSION >= 1

    def test_hard_fillers_canonical_set(self) -> None:
        # PodcastFillers-derived; covers ~96% of empirically annotated fillers.
        # Excludes "mm"/"mhm" (agreement sounds) and "you know"/"like"/"I mean"
        # (context-dependent, removal risks meaning loss).
        assert HARD_FILLERS == frozenset({"uh", "um", "uhm", "er", "ah"})


class TestCleanIdentity:
    def test_clean_returns_transcript(self, short_solo_transcript: Transcript) -> None:
        result = clean(short_solo_transcript)
        assert isinstance(result, Transcript)
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_clean_skeleton.py -v
```

Expected: FAIL — `yt2md.stages.clean` does not exist.

- [ ] **Step 3: Create stages package and clean.py**

`src/yt2md/stages/__init__.py`:

```python
"""Pipeline stages. Each module is one stage with a single public function."""
```

`src/yt2md/stages/clean.py`:

```python
"""Deterministic transcript cleaning.

Removes a fixed set of hard filler words (preserving surviving timestamps),
applies the 95% duration-weighted speaker-collapse rule, and drops speakers
contributing <1% of total duration as noise.

CLEANER_VERSION participates in the cleaned-artifact cache key. Bump on any
behavior change.
"""

from __future__ import annotations

from yt2md.models import Transcript

CLEANER_VERSION = 1

# Derived from PodcastFillers (Zhu et al. 2022). Covers ~96% of annotated fillers
# in podcast audio. Excludes "mm"/"mhm" (agreement sounds, non-filler) and
# "you know"/"like"/"I mean" (context-dependent; removal risks meaning loss).
HARD_FILLERS = frozenset({"uh", "um", "uhm", "er", "ah"})


def clean(transcript: Transcript) -> Transcript:
    """Pure function: returns a cleaned copy of the transcript.

    No-op skeleton — concrete behavior added in subsequent tasks.
    """
    return transcript
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_clean_skeleton.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/__init__.py src/yt2md/stages/clean.py tests/unit/test_clean_skeleton.py
git commit -m "$(cat <<'EOF'
feat(clean): add stages package, CLEANER_VERSION, HARD_FILLERS

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task C.2: Filler word removal preserves timestamps

**Files:**
- Create: `tests/unit/test_clean_fillers.py`
- Modify: `src/yt2md/stages/clean.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_clean_fillers.py`:

```python
"""Tests for filler-word removal in the clean stage."""

from yt2md.models import Segment, Transcript, Word
from yt2md.stages.clean import clean


def _word(text: str, start: float, end: float, speaker: str | None = "S0") -> Word:
    return Word(text=text, start=start, end=end, speaker=speaker)


def _segment(words: list[Word], speaker: str | None = "S0") -> Segment:
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(w.text for w in words),
        speaker=speaker,
        words=words,
    )


def _transcript(segments: list[Segment], speakers: list[str]) -> Transcript:
    duration = segments[-1].end if segments else 0.0
    return Transcript(
        language="en",
        duration_s=duration,
        backend="openai_transcribe",
        model_id="gpt-4o-transcribe",
        chunked=False,
        segments=segments,
        speakers=speakers,
    )


class TestFillerRemoval:
    def test_uh_dropped(self) -> None:
        t = _transcript(
            [_segment([_word("Uh", 0.0, 0.2), _word("hello", 0.3, 1.0)])],
            ["S0"],
        )
        result = clean(t)
        words = result.segments[0].words
        assert [w.text for w in words] == ["hello"]

    def test_uh_with_comma_dropped(self) -> None:
        t = _transcript(
            [_segment([_word("Uh,", 0.0, 0.2), _word("hello", 0.3, 1.0)])],
            ["S0"],
        )
        result = clean(t)
        words = result.segments[0].words
        assert [w.text for w in words] == ["hello"]

    def test_um_dropped(self) -> None:
        t = _transcript([_segment([_word("um", 0.0, 0.2), _word("yes", 0.3, 1.0)])], ["S0"])
        result = clean(t)
        assert [w.text for w in result.segments[0].words] == ["yes"]

    def test_all_hard_fillers_dropped(self) -> None:
        words = [
            _word("uh", 0.0, 0.1),
            _word("um", 0.2, 0.3),
            _word("er", 0.4, 0.5),
            _word("ah", 0.6, 0.7),
            _word("uhm", 0.8, 0.9),
            _word("hello", 1.0, 1.5),
        ]
        t = _transcript([_segment(words)], ["S0"])
        result = clean(t)
        assert [w.text for w in result.segments[0].words] == ["hello"]

    def test_like_preserved(self) -> None:
        # "like" is intentionally NOT a hard filler.
        t = _transcript(
            [_segment([_word("like", 0.0, 0.3), _word("water", 0.4, 1.0)])],
            ["S0"],
        )
        result = clean(t)
        assert [w.text for w in result.segments[0].words] == ["like", "water"]

    def test_mm_preserved(self) -> None:
        # "mm" is an agreement sound, not a filler.
        t = _transcript([_segment([_word("Mm.", 0.0, 0.5)])], ["S0"])
        result = clean(t)
        assert len(result.segments) == 1
        assert result.segments[0].words[0].text == "Mm."

    def test_surviving_word_timestamps_unchanged(self) -> None:
        t = _transcript(
            [_segment([
                _word("Uh", 0.0, 0.2),
                _word("dopamine", 0.5, 1.5),
                _word("signals", 1.6, 2.4),
            ])],
            ["S0"],
        )
        result = clean(t)
        words = result.segments[0].words
        assert words[0].text == "dopamine"
        assert words[0].start == 0.5
        assert words[0].end == 1.5
        assert words[1].start == 1.6
        assert words[1].end == 2.4

    def test_segment_text_rebuilt_from_surviving_words(self) -> None:
        t = _transcript(
            [_segment([
                _word("Uh,", 0.0, 0.2),
                _word("dopamine", 0.5, 1.5),
                _word("signals", 1.6, 2.4),
            ])],
            ["S0"],
        )
        result = clean(t)
        assert result.segments[0].text == "dopamine signals"
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_clean_fillers.py -v
```

Expected: tests FAIL (current `clean` is identity).

- [ ] **Step 3: Implement filler removal in `clean.py`**

Replace `clean()` and add a helper:

```python
import string


def clean(transcript: Transcript) -> Transcript:
    """Pure function: returns a cleaned copy of the transcript."""
    cleaned_segments = [_clean_segment(s) for s in transcript.segments]
    cleaned_segments = [s for s in cleaned_segments if s is not None]
    return Transcript(
        language=transcript.language,
        duration_s=transcript.duration_s,
        backend=transcript.backend,
        model_id=transcript.model_id,
        chunked=transcript.chunked,
        segments=cleaned_segments,
        speakers=transcript.speakers,
    )


def _clean_segment(segment: object) -> object:
    # segment is a Segment; using object here so callers see a stable signature
    # for the next task (speaker collapse) that will refine this.
    from yt2md.models import Segment, Word

    seg: Segment = segment  # type: ignore[assignment]
    kept: list[Word] = [w for w in seg.words if not _is_filler(w.text)]
    if not kept:
        return None
    rebuilt_text = " ".join(w.text for w in kept)
    return Segment(
        start=seg.start,
        end=seg.end,
        text=rebuilt_text,
        speaker=seg.speaker,
        words=kept,
    )


def _is_filler(token: str) -> bool:
    normalized = token.lower().strip(string.punctuation + string.whitespace)
    return normalized in HARD_FILLERS
```

(The `object`-typed inner helper is a temporary scaffold; clean it up in the next task when we add speaker collapse.)

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_clean_fillers.py -v
```

Expected: 8 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

Run lint — Ruff will complain about the `object` typing trick. Fix by importing `Segment` at the top of the file:

```python
from yt2md.models import Segment, Transcript, Word
```

And tighten `_clean_segment`:

```python
def _clean_segment(segment: Segment) -> Segment | None:
    kept: list[Word] = [w for w in segment.words if not _is_filler(w.text)]
    if not kept:
        return None
    rebuilt_text = " ".join(w.text for w in kept)
    return Segment(
        start=segment.start,
        end=segment.end,
        text=rebuilt_text,
        speaker=segment.speaker,
        words=kept,
    )
```

Re-run tests and lint:

```bash
uv run pytest tests/unit/test_clean_fillers.py -v
uv run ruff check src/ tests/
uv run mypy --strict src/
```

All pass. Commit:

```bash
git add src/yt2md/stages/clean.py tests/unit/test_clean_fillers.py
git commit -m "$(cat <<'EOF'
feat(clean): drop HARD_FILLERS words, preserve surviving timestamps

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task C.3: Speaker collapse at 95% duration-weighted threshold

**Files:**
- Create: `tests/unit/test_clean_speaker_collapse.py`
- Modify: `src/yt2md/stages/clean.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_clean_speaker_collapse.py`:

```python
"""Tests for the 95% speaker-collapse rule in the clean stage."""

from yt2md.models import Segment, Transcript, Word
from yt2md.stages.clean import clean


def _word(text: str, start: float, end: float, speaker: str) -> Word:
    return Word(text=text, start=start, end=end, speaker=speaker)


def _seg(start: float, end: float, speaker: str) -> Segment:
    word = _word("x", start, end, speaker)
    return Segment(start=start, end=end, text="x", speaker=speaker, words=[word])


class TestSpeakerCollapse:
    def test_collapse_at_96_percent(self) -> None:
        # SPEAKER_00 = 96s, SPEAKER_01 = 4s → 0.96 ≥ 0.95 → collapse.
        t = Transcript(
            language="en",
            duration_s=100.0,
            backend="openai_transcribe",
            model_id="gpt-4o-transcribe",
            chunked=False,
            segments=[
                _seg(0.0, 96.0, "SPEAKER_00"),
                _seg(96.0, 100.0, "SPEAKER_01"),
            ],
            speakers=["SPEAKER_00", "SPEAKER_01"],
        )
        result = clean(t)
        # Below 1% rule: SPEAKER_01 has 4% so survives the noise filter,
        # but the 95% collapse rule rewrites all to dominant.
        assert result.speakers == ["SPEAKER_00"]
        assert all(s.speaker == "SPEAKER_00" for s in result.segments)
        assert all(w.speaker == "SPEAKER_00" for s in result.segments for w in s.words)

    def test_no_collapse_at_94_percent(self) -> None:
        # SPEAKER_00 = 94s, SPEAKER_01 = 6s → 0.94 < 0.95 → no collapse.
        t = Transcript(
            language="en",
            duration_s=100.0,
            backend="openai_transcribe",
            model_id="gpt-4o-transcribe",
            chunked=False,
            segments=[
                _seg(0.0, 94.0, "SPEAKER_00"),
                _seg(94.0, 100.0, "SPEAKER_01"),
            ],
            speakers=["SPEAKER_00", "SPEAKER_01"],
        )
        result = clean(t)
        assert set(result.speakers) == {"SPEAKER_00", "SPEAKER_01"}

    def test_collapse_at_exactly_95_percent(self) -> None:
        # SPEAKER_00 = 95s, SPEAKER_01 = 5s → 0.95 ≥ 0.95 → collapse (≥, not >).
        t = Transcript(
            language="en",
            duration_s=100.0,
            backend="openai_transcribe",
            model_id="gpt-4o-transcribe",
            chunked=False,
            segments=[
                _seg(0.0, 95.0, "SPEAKER_00"),
                _seg(95.0, 100.0, "SPEAKER_01"),
            ],
            speakers=["SPEAKER_00", "SPEAKER_01"],
        )
        result = clean(t)
        assert result.speakers == ["SPEAKER_00"]

    def test_noise_speaker_dropped(self) -> None:
        # SPEAKER_00 = 99.5s, SPEAKER_01 = 0.5s → 0.5% → noise, drop segment.
        # SPEAKER_00 dominance = 0.995 → also collapse, but segment for SPEAKER_01
        # should be entirely removed (not just relabeled).
        t = Transcript(
            language="en",
            duration_s=100.0,
            backend="openai_transcribe",
            model_id="gpt-4o-transcribe",
            chunked=False,
            segments=[
                _seg(0.0, 99.5, "SPEAKER_00"),
                _seg(99.5, 100.0, "SPEAKER_01"),
            ],
            speakers=["SPEAKER_00", "SPEAKER_01"],
        )
        result = clean(t)
        assert result.speakers == ["SPEAKER_00"]
        # SPEAKER_01 contributed <1% → segment dropped, not relabeled.
        assert len(result.segments) == 1
        assert result.segments[0].speaker == "SPEAKER_00"

    def test_undiarized_unchanged(self) -> None:
        # No speakers → no collapse logic applies.
        t = Transcript(
            language="en",
            duration_s=10.0,
            backend="local_whisper",
            model_id="faster-whisper-medium",
            chunked=False,
            segments=[
                Segment(
                    start=0.0,
                    end=10.0,
                    text="hi",
                    speaker=None,
                    words=[Word(text="hi", start=0.0, end=10.0, speaker=None)],
                ),
            ],
            speakers=[],
        )
        result = clean(t)
        assert result.speakers == []
        assert result.segments[0].speaker is None
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_clean_speaker_collapse.py -v
```

Expected: tests FAIL (no collapse logic yet).

- [ ] **Step 3: Implement collapse + noise filtering**

Update `src/yt2md/stages/clean.py`. Add constants near the existing ones:

```python
COLLAPSE_THRESHOLD = 0.95
NOISE_THRESHOLD = 0.01
```

Refactor `clean()`:

```python
def clean(transcript: Transcript) -> Transcript:
    """Pure function: returns a cleaned copy of the transcript.

    Steps:
      1. Drop HARD_FILLERS words from every segment; drop empty segments.
      2. Compute per-speaker duration (sum of word.end - word.start).
      3. Drop segments whose speaker contributes <1% of total duration (noise).
      4. If the dominant speaker holds ≥95% of remaining duration, collapse:
         rewrite all word.speaker and segment.speaker to the dominant label,
         set transcript.speakers = [dominant].
    """
    filler_dropped = [_clean_segment(s) for s in transcript.segments]
    surviving = [s for s in filler_dropped if s is not None]

    if not transcript.speakers:
        return _replace_segments(transcript, surviving)

    per_speaker = _per_speaker_duration(surviving)
    total = sum(per_speaker.values())
    if total == 0:
        return _replace_segments(transcript, surviving)

    above_noise = {sp: d for sp, d in per_speaker.items() if d / total >= NOISE_THRESHOLD}
    surviving = [s for s in surviving if s.speaker in above_noise]

    if not surviving:
        return _replace_segments(transcript, surviving, speakers=[])

    dominant = max(above_noise, key=lambda sp: above_noise[sp])
    new_total = sum(above_noise.values())
    if above_noise[dominant] / new_total >= COLLAPSE_THRESHOLD:
        surviving = [_relabel_segment(s, dominant) for s in surviving]
        return _replace_segments(transcript, surviving, speakers=[dominant])

    return _replace_segments(
        transcript,
        surviving,
        speakers=sorted(above_noise.keys()),
    )


def _per_speaker_duration(segments: list[Segment]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for s in segments:
        for w in s.words:
            if w.speaker is None:
                continue
            durations[w.speaker] = durations.get(w.speaker, 0.0) + (w.end - w.start)
    return durations


def _relabel_segment(segment: Segment, label: str) -> Segment:
    relabeled_words = [
        Word(text=w.text, start=w.start, end=w.end, speaker=label) for w in segment.words
    ]
    return Segment(
        start=segment.start,
        end=segment.end,
        text=segment.text,
        speaker=label,
        words=relabeled_words,
    )


def _replace_segments(
    transcript: Transcript,
    segments: list[Segment],
    *,
    speakers: list[str] | None = None,
) -> Transcript:
    return Transcript(
        language=transcript.language,
        duration_s=transcript.duration_s,
        backend=transcript.backend,
        model_id=transcript.model_id,
        chunked=transcript.chunked,
        segments=segments,
        speakers=speakers if speakers is not None else transcript.speakers,
    )
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_clean_speaker_collapse.py -v
uv run pytest tests/unit/test_clean_fillers.py -v
```

Expected: all PASS (filler tests still green; new collapse tests green).

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/clean.py tests/unit/test_clean_speaker_collapse.py
git commit -m "$(cat <<'EOF'
feat(clean): 95% duration-weighted speaker collapse + 1% noise drop

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task C.4: Property test — surviving timestamps are a subset

**Files:**
- Create: `tests/unit/test_clean_property.py`

- [ ] **Step 1: Write the property test**

`tests/unit/test_clean_property.py`:

```python
"""Property test: clean() never invents timestamps.

Every surviving word's (start, end) tuple must have been in the input transcript.
"""

from hypothesis import given
from hypothesis import strategies as st

from yt2md.models import Segment, Transcript, Word
from yt2md.stages.clean import clean

_word_strategy = st.builds(
    Word,
    text=st.text(min_size=1, max_size=10).filter(lambda s: not s.isspace()),
    start=st.floats(min_value=0.0, max_value=100.0),
    end=st.floats(min_value=0.0, max_value=100.0),
    speaker=st.sampled_from(["S0", "S1", None]),
).filter(lambda w: w.end >= w.start)


def _segments_from_words(words: list[Word]) -> list[Segment]:
    if not words:
        return []
    return [
        Segment(
            start=words[0].start,
            end=max(w.end for w in words),
            text=" ".join(w.text for w in words),
            speaker=words[0].speaker,
            words=words,
        ),
    ]


@given(words=st.lists(_word_strategy, min_size=1, max_size=20))
def test_surviving_timestamps_are_subset_of_input(words: list[Word]) -> None:
    segments = _segments_from_words(words)
    duration = max(w.end for w in words)
    t = Transcript(
        language="en",
        duration_s=duration,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=segments,
        speakers=["S0", "S1"] if any(w.speaker for w in words) else [],
    )
    result = clean(t)
    input_pairs = {(w.start, w.end) for w in words}
    output_pairs = {(w.start, w.end) for s in result.segments for w in s.words}
    assert output_pairs.issubset(input_pairs)
```

- [ ] **Step 2: Run — confirm passes**

```bash
uv run pytest tests/unit/test_clean_property.py -v
```

Expected: PASS. If FAIL, the cleaner is mutating timestamps somewhere.

- [ ] **Step 3: Lint + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add tests/unit/test_clean_property.py
git commit -m "$(cat <<'EOF'
test(clean): property — surviving timestamps subset of input

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section D: render.py + Jinja2 template

`render.py` is the largest module in Phase 2. It builds the final markdown from a `StructuredDoc` and a cleaned `Transcript`. The Jinja2 template handles the document shape; render.py handles preprocessing (paragraph grouping, name substitution, URL building).

---

### Task D.1: Jinja2 environment + render() skeleton

**Files:**
- Create: `tests/unit/test_render_skeleton.py`
- Create: `src/yt2md/stages/render.py`
- Create: `src/yt2md/templates/document.md.j2`
- Create: `src/yt2md/templates/__init__.py` (empty marker)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_render_skeleton.py`:

```python
"""Smoke test: render() returns a non-empty string given a minimal StructuredDoc."""

from datetime import date

import pytest

from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    Frontmatter,
    StructuredDoc,
    Takeaway,
    Transcript,
)
from yt2md.stages.render import render


@pytest.fixture
def minimal_doc() -> StructuredDoc:
    fm = Frontmatter(
        title="Test",
        channel="TC",
        url="https://www.youtube.com/watch?v=v",
        video_id="v",
        published=date(2025, 1, 1),
        duration_seconds=60,
        captured_at=date(2026, 5, 23),
        schema_version=CURRENT_SCHEMA_VERSION,
        genre="podcast",
        speakers=["Alice"],
        topics=[],
        people_mentioned=[],
        works_mentioned=[],
    )
    return StructuredDoc(
        frontmatter=fm,
        tldr="A short TLDR sentence.",
        takeaways=[Takeaway(text="One.", timestamp_s=0.0)],
        concepts=[],
        references=[],
        quotes=[],
        sections=[],
        open_questions=[],
        speaker_name_map={"SPEAKER_00": "Alice"},
    )


@pytest.fixture
def empty_transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_s=60.0,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=[],
        speakers=["Alice"],
    )


def test_render_returns_string(minimal_doc: StructuredDoc, empty_transcript: Transcript) -> None:
    md = render(minimal_doc, empty_transcript)
    assert isinstance(md, str)
    assert md.strip()


def test_render_starts_with_frontmatter(minimal_doc: StructuredDoc, empty_transcript: Transcript) -> None:
    md = render(minimal_doc, empty_transcript)
    assert md.startswith("---\n")


def test_render_contains_title(minimal_doc: StructuredDoc, empty_transcript: Transcript) -> None:
    md = render(minimal_doc, empty_transcript)
    assert "Test" in md


def test_render_contains_tldr(minimal_doc: StructuredDoc, empty_transcript: Transcript) -> None:
    md = render(minimal_doc, empty_transcript)
    assert "A short TLDR sentence." in md
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_render_skeleton.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write the Jinja template**

`src/yt2md/templates/document.md.j2`:

```jinja
---
title: {{ frontmatter.title | yaml_str }}
channel: {{ frontmatter.channel | yaml_str }}
url: {{ frontmatter.url }}
video_id: {{ frontmatter.video_id }}
published: {{ frontmatter.published }}
duration_seconds: {{ frontmatter.duration_seconds }}
captured_at: {{ frontmatter.captured_at }}
schema_version: {{ frontmatter.schema_version }}
genre: {{ frontmatter.genre }}
speakers: {{ frontmatter.speakers | yaml_list }}
topics: {{ frontmatter.topics | yaml_list }}
people_mentioned: {{ frontmatter.people_mentioned | yaml_list }}
works_mentioned: {{ frontmatter.works_mentioned | yaml_list }}
---

# {{ frontmatter.title }}

## TL;DR

{{ tldr }}
```

(Subsequent tasks extend the template — Takeaways, Concepts, References, Quotes, Detailed Notes, Open Questions, Full Transcript.)

- [ ] **Step 4: Write render.py**

`src/yt2md/stages/render.py`:

```python
"""Render a StructuredDoc + Transcript to the final markdown document.

The Jinja2 template owns the document shape; this module owns preprocessing
(YAML-safe filters, paragraph grouping, name substitution, URL building).
"""

from __future__ import annotations

import json
from importlib import resources

from jinja2 import Environment, FileSystemLoader, select_autoescape

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
    return template.render(
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
```

- [ ] **Step 5: Update pyproject so templates ship with the package**

Modify `pyproject.toml` `[tool.hatch.build.targets.wheel]` section to:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/yt2md"]

[tool.hatch.build.targets.wheel.force-include]
"src/yt2md/templates/document.md.j2" = "yt2md/templates/document.md.j2"
```

- [ ] **Step 6: Create templates __init__ marker**

```bash
touch src/yt2md/templates/__init__.py
```

(Empty file. Makes the templates directory importable as a package so `resources.files("yt2md") / "templates"` works.)

- [ ] **Step 7: Run — confirm passes**

```bash
uv run pytest tests/unit/test_render_skeleton.py -v
```

Expected: 4 PASS.

- [ ] **Step 8: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/render.py src/yt2md/templates/document.md.j2 src/yt2md/templates/__init__.py pyproject.toml tests/unit/test_render_skeleton.py
git commit -m "$(cat <<'EOF'
feat(render): add Jinja2-based render() with frontmatter + TL;DR skeleton

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task D.2: Render Key Takeaways with timestamp deep-links

**Files:**
- Create: `tests/unit/test_render_takeaways.py`
- Modify: `src/yt2md/templates/document.md.j2`
- Modify: `src/yt2md/stages/render.py` (add URL builder)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_render_takeaways.py`:

```python
"""Tests for Key Takeaways section rendering with timestamp deep-links."""

from datetime import date

import pytest

from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    Frontmatter,
    StructuredDoc,
    Takeaway,
    Transcript,
)
from yt2md.stages.render import render


@pytest.fixture
def doc_with_takeaways() -> StructuredDoc:
    fm = Frontmatter(
        title="Test",
        channel="TC",
        url="https://www.youtube.com/watch?v=abc",
        video_id="abc",
        published=date(2025, 1, 1),
        duration_seconds=600,
        captured_at=date(2026, 5, 23),
        schema_version=CURRENT_SCHEMA_VERSION,
        genre="podcast",
        speakers=["Alice"],
        topics=[],
        people_mentioned=[],
        works_mentioned=[],
    )
    return StructuredDoc(
        frontmatter=fm,
        tldr="t",
        takeaways=[
            Takeaway(text="Dopamine signals anticipation.", timestamp_s=252.0),
            Takeaway(text="It peaks before reward.", timestamp_s=510.5),
        ],
        concepts=[],
        references=[],
        quotes=[],
        sections=[],
        open_questions=[],
        speaker_name_map={},
    )


@pytest.fixture
def empty_transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_s=600.0,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=[],
        speakers=["Alice"],
    )


def test_section_header_present(doc_with_takeaways: StructuredDoc, empty_transcript: Transcript) -> None:
    md = render(doc_with_takeaways, empty_transcript)
    assert "## Key Takeaways" in md


def test_takeaway_text_present(doc_with_takeaways: StructuredDoc, empty_transcript: Transcript) -> None:
    md = render(doc_with_takeaways, empty_transcript)
    assert "Dopamine signals anticipation." in md


def test_timestamp_link_format(doc_with_takeaways: StructuredDoc, empty_transcript: Transcript) -> None:
    md = render(doc_with_takeaways, empty_transcript)
    # 252s → 04:12 display, &t=252s URL
    assert "[04:12]" in md
    assert "https://www.youtube.com/watch?v=abc&t=252s" in md


def test_fractional_timestamp_truncated_to_int_seconds(
    doc_with_takeaways: StructuredDoc, empty_transcript: Transcript
) -> None:
    md = render(doc_with_takeaways, empty_transcript)
    # 510.5s → int(510) → 08:30
    assert "[08:30]" in md
    assert "&t=510s" in md
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_render_takeaways.py -v
```

Expected: header / link tests FAIL.

- [ ] **Step 3: Add URL/timestamp helpers to render.py**

Append helpers below `_yaml_list`:

```python
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
```

Register them in `_build_env`:

```python
    env.filters["yaml_str"] = _yaml_str
    env.filters["yaml_list"] = _yaml_list
    env.filters["mmss"] = _mmss
    env.globals["ytlink"] = _ytlink
```

- [ ] **Step 4: Extend the Jinja template**

Append to `src/yt2md/templates/document.md.j2`:

```jinja

## Key Takeaways

{% for t in takeaways -%}
- {{ t.text }} [[{{ t.timestamp_s | mmss }}]]({{ ytlink(frontmatter.url, t.timestamp_s) }})
{% endfor %}
```

- [ ] **Step 5: Run — confirm passes**

```bash
uv run pytest tests/unit/test_render_takeaways.py -v
uv run pytest tests/unit/test_render_skeleton.py -v
```

Expected: all PASS.

- [ ] **Step 6: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/render.py src/yt2md/templates/document.md.j2 tests/unit/test_render_takeaways.py
git commit -m "$(cat <<'EOF'
feat(render): render Key Takeaways with mm:ss and ytlink deep-links

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task D.3: Render Concepts, References (with emoji prefixes), Quotes

**Files:**
- Create: `tests/unit/test_render_sections.py`
- Modify: `src/yt2md/templates/document.md.j2`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_render_sections.py`:

```python
"""Tests for Concepts, References (with emoji), Notable Quotes sections."""

from datetime import date

import pytest

from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    Concept,
    Frontmatter,
    Quote,
    Reference,
    StructuredDoc,
    Transcript,
)
from yt2md.stages.render import render


def _doc(
    concepts: list[Concept] | None = None,
    references: list[Reference] | None = None,
    quotes: list[Quote] | None = None,
) -> StructuredDoc:
    fm = Frontmatter(
        title="T",
        channel="C",
        url="https://www.youtube.com/watch?v=v",
        video_id="v",
        published=date(2025, 1, 1),
        duration_seconds=60,
        captured_at=date(2026, 5, 23),
        schema_version=CURRENT_SCHEMA_VERSION,
        genre="podcast",
        speakers=["A"],
        topics=[],
        people_mentioned=[],
        works_mentioned=[],
    )
    return StructuredDoc(
        frontmatter=fm,
        tldr="t",
        takeaways=[],
        concepts=concepts or [],
        references=references or [],
        quotes=quotes or [],
        sections=[],
        open_questions=[],
        speaker_name_map={},
    )


@pytest.fixture
def empty_transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_s=60.0,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=[],
        speakers=["A"],
    )


class TestConcepts:
    def test_concept_rendered(self, empty_transcript: Transcript) -> None:
        d = _doc(concepts=[
            Concept(name="Reward Prediction Error", definition="Gap between expected and actual.", timestamp_s=510.0),
        ])
        md = render(d, empty_transcript)
        assert "## Concepts & Definitions" in md
        assert "Reward Prediction Error" in md
        assert "Gap between expected and actual." in md
        assert "[08:30]" in md


class TestReferences:
    @pytest.mark.parametrize(
        ("kind", "expected_emoji"),
        [
            ("book", "📚"),
            ("paper", "📄"),
            ("person", "👤"),
            ("tool", "🛠"),
            ("video", "🎬"),
            ("other", "🔗"),
        ],
    )
    def test_emoji_prefix(self, empty_transcript: Transcript, kind: str, expected_emoji: str) -> None:
        d = _doc(references=[
            Reference(kind=kind, name="X", context="c", timestamp_s=0.0),  # type: ignore[arg-type]
        ])
        md = render(d, empty_transcript)
        assert expected_emoji in md

    def test_reference_text(self, empty_transcript: Transcript) -> None:
        d = _doc(references=[
            Reference(kind="book", name="The Molecule of More", context="Cited as accessible primer", timestamp_s=902.0),
        ])
        md = render(d, empty_transcript)
        assert "## References Mentioned" in md
        assert "The Molecule of More" in md
        assert "Cited as accessible primer" in md


class TestQuotes:
    def test_quote_rendered(self, empty_transcript: Transcript) -> None:
        d = _doc(quotes=[
            Quote(text="Pursuit, not pleasure.", speaker="Andrew Huberman", timestamp_s=754.0),
        ])
        md = render(d, empty_transcript)
        assert "## Notable Quotes" in md
        # Block-quote prefix
        assert "> Pursuit, not pleasure." in md
        assert "— Andrew Huberman" in md
        assert "[12:34]" in md
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_render_sections.py -v
```

Expected: FAIL.

- [ ] **Step 3: Extend the Jinja template**

Append to `src/yt2md/templates/document.md.j2`:

```jinja

{% if concepts %}
## Concepts & Definitions

{% for c in concepts -%}
**{{ c.name }}** — {{ c.definition }} [[{{ c.timestamp_s | mmss }}]]({{ ytlink(frontmatter.url, c.timestamp_s) }})

{% endfor %}
{% endif %}

{% if references %}
## References Mentioned

{% for r in references -%}
- {{ emoji_for(r.kind) }} **{{ r.name }}** — {{ r.context }} [[{{ r.timestamp_s | mmss }}]]({{ ytlink(frontmatter.url, r.timestamp_s) }})
{% endfor %}
{% endif %}

{% if quotes %}
## Notable Quotes

{% for q in quotes -%}
> {{ q.text }}
> — {{ q.speaker }}, [[{{ q.timestamp_s | mmss }}]]({{ ytlink(frontmatter.url, q.timestamp_s) }})

{% endfor %}
{% endif %}
```

- [ ] **Step 4: Add `emoji_for` helper to render.py**

Append below `_ytlink`:

```python
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
```

Register in `_build_env`:

```python
    env.globals["emoji_for"] = _emoji_for
```

- [ ] **Step 5: Run — confirm passes**

```bash
uv run pytest tests/unit/test_render_sections.py -v
uv run pytest tests/unit/test_render_takeaways.py tests/unit/test_render_skeleton.py -v
```

Expected: all PASS.

- [ ] **Step 6: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/render.py src/yt2md/templates/document.md.j2 tests/unit/test_render_sections.py
git commit -m "$(cat <<'EOF'
feat(render): add Concepts, References (emoji), Notable Quotes sections

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task D.4: Render Detailed Notes sections + Open Questions

**Files:**
- Create: `tests/unit/test_render_detailed.py`
- Modify: `src/yt2md/templates/document.md.j2`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_render_detailed.py`:

```python
"""Tests for Detailed Notes sections and Open Questions."""

from datetime import date

import pytest

from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    DetailedSection,
    Frontmatter,
    StructuredDoc,
    Transcript,
)
from yt2md.stages.render import render


def _doc(
    sections: list[DetailedSection] | None = None,
    open_questions: list[str] | None = None,
) -> StructuredDoc:
    fm = Frontmatter(
        title="T",
        channel="C",
        url="https://www.youtube.com/watch?v=v",
        video_id="v",
        published=date(2025, 1, 1),
        duration_seconds=60,
        captured_at=date(2026, 5, 23),
        schema_version=CURRENT_SCHEMA_VERSION,
        genre="podcast",
        speakers=["A"],
        topics=[],
        people_mentioned=[],
        works_mentioned=[],
    )
    return StructuredDoc(
        frontmatter=fm,
        tldr="t",
        takeaways=[],
        concepts=[],
        references=[],
        quotes=[],
        sections=sections or [],
        open_questions=open_questions or [],
        speaker_name_map={},
    )


@pytest.fixture
def empty_transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_s=60.0,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=[],
        speakers=["A"],
    )


class TestDetailedSections:
    def test_header_present(self, empty_transcript: Transcript) -> None:
        d = _doc(sections=[
            DetailedSection(heading="What dopamine does", body="Huberman explains...", timestamp_s=0.0),
        ])
        md = render(d, empty_transcript)
        assert "## Detailed Notes" in md

    def test_subheading_with_timestamp(self, empty_transcript: Transcript) -> None:
        d = _doc(sections=[
            DetailedSection(heading="Tools to raise dopamine", body="Cold exposure...", timestamp_s=2720.0),
        ])
        md = render(d, empty_transcript)
        assert "### Tools to raise dopamine" in md
        assert "[45:20]" in md
        assert "&t=2720s" in md

    def test_body_present(self, empty_transcript: Transcript) -> None:
        d = _doc(sections=[
            DetailedSection(heading="H", body="Multi-paragraph body content.", timestamp_s=0.0),
        ])
        md = render(d, empty_transcript)
        assert "Multi-paragraph body content." in md


class TestOpenQuestions:
    def test_header_present(self, empty_transcript: Transcript) -> None:
        d = _doc(open_questions=["What about D1 vs D2 receptors?"])
        md = render(d, empty_transcript)
        assert "## Open Questions" in md

    def test_questions_bullet(self, empty_transcript: Transcript) -> None:
        d = _doc(open_questions=["Q1?", "Q2?"])
        md = render(d, empty_transcript)
        assert "- Q1?" in md
        assert "- Q2?" in md

    def test_section_omitted_when_empty(self, empty_transcript: Transcript) -> None:
        d = _doc()
        md = render(d, empty_transcript)
        assert "## Open Questions" not in md
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_render_detailed.py -v
```

Expected: FAIL.

- [ ] **Step 3: Extend the Jinja template**

Append to `src/yt2md/templates/document.md.j2`:

```jinja

{% if sections %}
## Detailed Notes

{% for s in sections %}
### {{ s.heading }} [[{{ s.timestamp_s | mmss }}]]({{ ytlink(frontmatter.url, s.timestamp_s) }})

{{ s.body }}

{% endfor %}
{% endif %}

{% if open_questions %}
## Open Questions

{% for q in open_questions -%}
- {{ q }}
{% endfor %}
{% endif %}
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_render_detailed.py -v
uv run pytest tests/unit/test_render_skeleton.py tests/unit/test_render_takeaways.py tests/unit/test_render_sections.py -v
```

Expected: all PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/templates/document.md.j2 tests/unit/test_render_detailed.py
git commit -m "$(cat <<'EOF'
feat(render): add Detailed Notes sections and Open Questions

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task D.5: Full Transcript section with ~60s paragraph grouping and speaker substitution

**Files:**
- Create: `tests/unit/test_render_transcript.py`
- Modify: `src/yt2md/stages/render.py`
- Modify: `src/yt2md/templates/document.md.j2`

This is the heaviest task in render — paragraph grouping by 60s blocks + speaker change forces new paragraph + SPEAKER_NN name substitution.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_render_transcript.py`:

```python
"""Tests for the Full Cleaned Transcript section.

Requirements:
  - Sections labeled ## Full Cleaned Transcript
  - Words grouped into ~60s paragraphs with [mm:ss] timestamp markers at block start
  - Speaker change forces a new paragraph (even if under 60s)
  - SPEAKER_NN labels substituted using doc.speaker_name_map
"""

from datetime import date

import pytest

from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    Frontmatter,
    Segment,
    StructuredDoc,
    Transcript,
    Word,
)
from yt2md.stages.render import render


def _doc(speaker_name_map: dict[str, str]) -> StructuredDoc:
    fm = Frontmatter(
        title="T",
        channel="C",
        url="https://www.youtube.com/watch?v=v",
        video_id="v",
        published=date(2025, 1, 1),
        duration_seconds=200,
        captured_at=date(2026, 5, 23),
        schema_version=CURRENT_SCHEMA_VERSION,
        genre="podcast",
        speakers=list(speaker_name_map.values()) or ["A"],
        topics=[],
        people_mentioned=[],
        works_mentioned=[],
    )
    return StructuredDoc(
        frontmatter=fm,
        tldr="t",
        takeaways=[],
        concepts=[],
        references=[],
        quotes=[],
        sections=[],
        open_questions=[],
        speaker_name_map=speaker_name_map,
    )


def _seg(start: float, end: float, text: str, speaker: str | None) -> Segment:
    words = [Word(text=text, start=start, end=end, speaker=speaker)]
    return Segment(start=start, end=end, text=text, speaker=speaker, words=words)


class TestTranscriptSection:
    def test_section_header(self) -> None:
        d = _doc({})
        t = Transcript(
            language="en",
            duration_s=10.0,
            backend="openai_transcribe",
            model_id="m",
            chunked=False,
            segments=[_seg(0.0, 10.0, "hello", None)],
            speakers=[],
        )
        md = render(d, t)
        assert "## Full Cleaned Transcript" in md

    def test_timestamp_marker_at_paragraph_start(self) -> None:
        d = _doc({})
        t = Transcript(
            language="en",
            duration_s=10.0,
            backend="openai_transcribe",
            model_id="m",
            chunked=False,
            segments=[_seg(0.0, 10.0, "hello", None)],
            speakers=[],
        )
        md = render(d, t)
        assert "**[00:00]**" in md


class TestSpeakerNameSubstitution:
    def test_speaker_name_substituted(self) -> None:
        d = _doc({"SPEAKER_00": "Andrew Huberman"})
        t = Transcript(
            language="en",
            duration_s=10.0,
            backend="openai_transcribe",
            model_id="m",
            chunked=False,
            segments=[_seg(0.0, 10.0, "Welcome.", "SPEAKER_00")],
            speakers=["SPEAKER_00"],
        )
        md = render(d, t)
        assert "Andrew Huberman: Welcome." in md
        assert "SPEAKER_00" not in md

    def test_unmapped_speaker_label_preserved(self) -> None:
        d = _doc({})  # no map
        t = Transcript(
            language="en",
            duration_s=10.0,
            backend="openai_transcribe",
            model_id="m",
            chunked=False,
            segments=[_seg(0.0, 10.0, "Welcome.", "SPEAKER_00")],
            speakers=["SPEAKER_00"],
        )
        md = render(d, t)
        assert "SPEAKER_00: Welcome." in md


class TestParagraphGrouping:
    def test_speaker_change_forces_new_paragraph(self) -> None:
        d = _doc({"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})
        t = Transcript(
            language="en",
            duration_s=20.0,
            backend="openai_transcribe",
            model_id="m",
            chunked=False,
            segments=[
                _seg(0.0, 5.0, "Hello.", "SPEAKER_00"),
                _seg(5.5, 10.0, "Hi back.", "SPEAKER_01"),
                _seg(10.5, 20.0, "And then.", "SPEAKER_00"),
            ],
            speakers=["SPEAKER_00", "SPEAKER_01"],
        )
        md = render(d, t)
        # Each speaker change → its own paragraph with its own [mm:ss] marker
        assert "**[00:00]** Alice: Hello." in md
        assert "**[00:05]** Bob: Hi back." in md
        assert "**[00:10]** Alice: And then." in md

    def test_same_speaker_under_60s_grouped(self) -> None:
        d = _doc({})
        t = Transcript(
            language="en",
            duration_s=30.0,
            backend="openai_transcribe",
            model_id="m",
            chunked=False,
            segments=[
                _seg(0.0, 10.0, "First.", None),
                _seg(10.0, 20.0, "Second.", None),
                _seg(20.0, 30.0, "Third.", None),
            ],
            speakers=[],
        )
        md = render(d, t)
        # All three should be in one paragraph (same speaker None, within 60s)
        assert "**[00:00]**" in md
        assert "**[00:10]**" not in md
        assert "**[00:20]**" not in md
        # Joined text
        assert "First. Second. Third." in md

    def test_60s_boundary_forces_new_paragraph(self) -> None:
        d = _doc({})
        t = Transcript(
            language="en",
            duration_s=120.0,
            backend="openai_transcribe",
            model_id="m",
            chunked=False,
            segments=[
                _seg(0.0, 50.0, "First.", None),
                _seg(50.0, 65.0, "Second.", None),  # crosses 60s
                _seg(65.0, 120.0, "Third.", None),
            ],
            speakers=[],
        )
        md = render(d, t)
        # Block 1: [00:00] First. Second.
        # Block 2: [01:05] Third.  (next segment after block 1 boundary)
        assert "**[00:00]** First. Second." in md
        assert "**[01:05]** Third." in md
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_render_transcript.py -v
```

Expected: FAIL.

- [ ] **Step 3: Add paragraph-grouping helper in render.py**

Append to `src/yt2md/stages/render.py`:

```python
from dataclasses import dataclass

from yt2md.models import Segment

PARAGRAPH_DURATION_S = 60.0


@dataclass(frozen=True)
class TranscriptParagraph:
    start_s: float
    speaker: str | None
    text: str


def _group_into_paragraphs(transcript: Transcript) -> list[TranscriptParagraph]:
    """Group segments into ~60-second paragraphs.

    Rules:
      - A new paragraph starts when speaker changes from the prior segment.
      - A new paragraph starts when the current segment's start is ≥ paragraph_start + 60s.
      - Within a paragraph, segment texts are space-joined.
    """
    paragraphs: list[TranscriptParagraph] = []
    current_start: float | None = None
    current_speaker: str | None = None
    current_texts: list[str] = []

    def _flush() -> None:
        if current_start is not None and current_texts:
            paragraphs.append(
                TranscriptParagraph(
                    start_s=current_start,
                    speaker=current_speaker,
                    text=" ".join(current_texts),
                )
            )

    for seg in transcript.segments:
        start_new = (
            current_start is None
            or seg.speaker != current_speaker
            or seg.start >= current_start + PARAGRAPH_DURATION_S
        )
        if start_new:
            _flush()
            current_start = seg.start
            current_speaker = seg.speaker
            current_texts = [seg.text]
        else:
            current_texts.append(seg.text)

    _flush()
    return paragraphs


def _resolve_speaker(label: str | None, name_map: dict[str, str]) -> str:
    if label is None:
        return ""
    return name_map.get(label, label)
```

Wire into `render()` — modify the template render call:

```python
def render(doc: StructuredDoc, transcript: Transcript) -> str:
    env = _build_env()
    paragraphs = _group_into_paragraphs(transcript)
    resolved_paragraphs = [
        {
            "start_s": p.start_s,
            "speaker": _resolve_speaker(p.speaker, doc.speaker_name_map),
            "text": p.text,
        }
        for p in paragraphs
    ]
    template = env.get_template("document.md.j2")
    return template.render(
        frontmatter=doc.frontmatter,
        tldr=doc.tldr,
        takeaways=doc.takeaways,
        concepts=doc.concepts,
        references=doc.references,
        quotes=doc.quotes,
        sections=doc.sections,
        open_questions=doc.open_questions,
        transcript_paragraphs=resolved_paragraphs,
    )
```

- [ ] **Step 4: Extend the Jinja template**

Append to `src/yt2md/templates/document.md.j2`:

```jinja

## Full Cleaned Transcript

{% for p in transcript_paragraphs %}
**[{{ p.start_s | mmss }}]**{% if p.speaker %} {{ p.speaker }}:{% endif %} {{ p.text }}

{% endfor %}
```

- [ ] **Step 5: Run — confirm passes**

```bash
uv run pytest tests/unit/test_render_transcript.py -v
uv run pytest tests/unit/test_render_skeleton.py tests/unit/test_render_takeaways.py tests/unit/test_render_sections.py tests/unit/test_render_detailed.py -v
```

Expected: all PASS.

- [ ] **Step 6: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/render.py src/yt2md/templates/document.md.j2 tests/unit/test_render_transcript.py
git commit -m "$(cat <<'EOF'
feat(render): Full Transcript with 60s paragraph grouping and name substitution

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task D.6: Golden end-to-end render test

**Files:**
- Create: `tests/fixtures/structured/sample_doc.json`
- Create: `tests/fixtures/markdown/sample_doc.md` (golden)
- Create: `tests/unit/test_render_golden.py`

This is the integration test for render. A pre-built `StructuredDoc` + a small `Transcript` round-trip to a known-byte-exact markdown.

- [ ] **Step 1: Build the fixture StructuredDoc and golden markdown together**

The golden file MUST match exactly what `render()` produces. Recommended workflow:
1. Write `sample_doc.json` with the inputs.
2. Run a one-off script to render it and capture the output as `sample_doc.md`.
3. Inspect the output for correctness against the spec's example.
4. Commit both.

`tests/fixtures/structured/sample_doc.json`:

```json
{
  "frontmatter": {
    "title": "Dopamine, Motivation & Drive",
    "channel": "Huberman Lab",
    "url": "https://www.youtube.com/watch?v=abc123",
    "video_id": "abc123",
    "published": "2024-03-15",
    "duration_seconds": 5025,
    "captured_at": "2026-05-23",
    "schema_version": 1,
    "genre": "podcast",
    "speakers": ["Andrew Huberman"],
    "topics": ["dopamine", "motivation"],
    "people_mentioned": ["Robert Sapolsky"],
    "works_mentioned": ["The Molecule of More"]
  },
  "tldr": "Huberman reframes dopamine as the molecule of pursuit.",
  "takeaways": [
    {"text": "Dopamine signals anticipation, not reward.", "timestamp_s": 252.0}
  ],
  "concepts": [
    {"name": "Reward Prediction Error", "definition": "Gap between expected and actual reward.", "timestamp_s": 510.0}
  ],
  "references": [
    {"kind": "book", "name": "The Molecule of More", "context": "Cited as accessible primer.", "timestamp_s": 902.0}
  ],
  "quotes": [
    {"text": "Pursuit, not pleasure.", "speaker": "Andrew Huberman", "timestamp_s": 754.0}
  ],
  "sections": [
    {"heading": "What dopamine does", "body": "Huberman opens by distinguishing tonic and phasic dopamine.", "timestamp_s": 0.0}
  ],
  "open_questions": ["D1 vs D2 receptor roles?"],
  "speaker_name_map": {"SPEAKER_00": "Andrew Huberman"}
}
```

- [ ] **Step 2: Generate the golden markdown**

Generate the golden by running a one-off render. Save to `tests/fixtures/markdown/sample_doc.md`:

```bash
uv run python -c "
import json
from pathlib import Path
from yt2md.models import StructuredDoc, Transcript, Segment, Word
from yt2md.stages.render import render

doc = StructuredDoc.model_validate(json.loads(
    Path('tests/fixtures/structured/sample_doc.json').read_text(encoding='utf-8')
))
# Minimal transcript for golden — one short segment
t = Transcript(
    language='en',
    duration_s=10.0,
    backend='openai_transcribe',
    model_id='gpt-4o-transcribe',
    chunked=False,
    segments=[Segment(
        start=0.0, end=10.0, text='Welcome to the show.',
        speaker='SPEAKER_00',
        words=[Word(text='Welcome', start=0.0, end=10.0, speaker='SPEAKER_00')],
    )],
    speakers=['SPEAKER_00'],
)
md = render(doc, t)
Path('tests/fixtures/markdown/sample_doc.md').write_text(md, encoding='utf-8')
print(md[:200])
"
```

Verify the output matches the spec's expected shape (frontmatter → h1 → TL;DR → sections in order). If anything looks wrong, fix the template, regenerate.

- [ ] **Step 3: Write the golden test**

`tests/unit/test_render_golden.py`:

```python
"""Golden test: render(fixture_doc, fixture_transcript) == fixture_markdown byte-for-byte."""

import json
from pathlib import Path

from yt2md.models import Segment, StructuredDoc, Transcript, Word
from yt2md.stages.render import render


def test_golden(fixtures_dir: Path) -> None:
    doc = StructuredDoc.model_validate(
        json.loads((fixtures_dir / "structured" / "sample_doc.json").read_text(encoding="utf-8"))
    )
    transcript = Transcript(
        language="en",
        duration_s=10.0,
        backend="openai_transcribe",
        model_id="gpt-4o-transcribe",
        chunked=False,
        segments=[
            Segment(
                start=0.0,
                end=10.0,
                text="Welcome to the show.",
                speaker="SPEAKER_00",
                words=[Word(text="Welcome", start=0.0, end=10.0, speaker="SPEAKER_00")],
            ),
        ],
        speakers=["SPEAKER_00"],
    )
    expected = (fixtures_dir / "markdown" / "sample_doc.md").read_text(encoding="utf-8")
    actual = render(doc, transcript)
    assert actual == expected
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_render_golden.py -v
```

Expected: PASS. If FAIL, the regenerated golden in Step 2 was stale — regenerate after every template change.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add tests/fixtures/structured/sample_doc.json tests/fixtures/markdown/sample_doc.md tests/unit/test_render_golden.py
git commit -m "$(cat <<'EOF'
test(render): add golden fixture + byte-exact end-to-end render test

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section E: write.py (filename + atomic write)

---

### Task E.1: slugify helper

**Files:**
- Create: `tests/unit/test_write_slugify.py`
- Create: `src/yt2md/stages/write.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_write_slugify.py`:

```python
"""Tests for slugify(): ASCII, hyphenated, max 80 chars."""

from yt2md.stages.write import slugify


class TestSlugify:
    def test_basic_lowercase(self) -> None:
        assert slugify("Hello World") == "hello-world"

    def test_strips_punctuation(self) -> None:
        assert slugify("Dopamine, Motivation & Drive!") == "dopamine-motivation-drive"

    def test_collapses_whitespace(self) -> None:
        assert slugify("hello   world   here") == "hello-world-here"

    def test_max_80_chars(self) -> None:
        s = "a" * 100
        out = slugify(s)
        assert len(out) <= 80

    def test_ascii_only(self) -> None:
        # Unicode chars stripped (or transliterated to ASCII-friendly)
        out = slugify("café résumé")
        assert out.isascii()

    def test_no_leading_or_trailing_hyphens(self) -> None:
        out = slugify("---hello---")
        assert not out.startswith("-")
        assert not out.endswith("-")

    def test_no_consecutive_hyphens(self) -> None:
        out = slugify("hello  --  world")
        assert "--" not in out

    def test_empty_input_returns_empty(self) -> None:
        assert slugify("") == ""

    def test_only_punctuation_returns_empty(self) -> None:
        assert slugify("!@#$%^&*()") == ""
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_write_slugify.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write `src/yt2md/stages/write.py`**

```python
"""Write stage: build deterministic filename + atomically write the markdown."""

from __future__ import annotations

import re
import unicodedata

MAX_SLUG_LENGTH = 80


def slugify(text: str) -> str:
    """Lowercase, ASCII, hyphenated, ≤80 chars. Empty input → empty output.

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
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_write_slugify.py -v
```

Expected: 9 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/write.py tests/unit/test_write_slugify.py
git commit -m "$(cat <<'EOF'
feat(write): add slugify() with ASCII normalization and 80-char cap

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task E.2: build_filename from frontmatter

**Files:**
- Create: `tests/unit/test_write_filename.py`
- Modify: `src/yt2md/stages/write.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_write_filename.py`:

```python
"""Tests for build_filename: {published}__{channel-slug}__{title-slug}.md format."""

from datetime import date

from yt2md.models import CURRENT_SCHEMA_VERSION, Frontmatter
from yt2md.stages.write import build_filename


def _fm(title: str = "Title", channel: str = "Channel", published: date | None = None) -> Frontmatter:
    return Frontmatter(
        title=title,
        channel=channel,
        url="u",
        video_id="vid",
        published=published or date(2024, 3, 15),
        duration_seconds=10,
        captured_at=date(2026, 5, 23),
        schema_version=CURRENT_SCHEMA_VERSION,
        genre="podcast",
        speakers=[],
        topics=[],
        people_mentioned=[],
        works_mentioned=[],
    )


class TestBuildFilename:
    def test_format(self) -> None:
        fm = _fm(title="Dopamine, Motivation & Drive", channel="Huberman Lab")
        assert build_filename(fm) == "2024-03-15__huberman-lab__dopamine-motivation-drive.md"

    def test_uses_published_not_captured(self) -> None:
        fm = _fm(published=date(2020, 1, 1))
        out = build_filename(fm)
        assert out.startswith("2020-01-01__")

    def test_slugifies_channel(self) -> None:
        fm = _fm(channel="The Tim Ferriss Show!")
        out = build_filename(fm)
        assert "__the-tim-ferriss-show__" in out

    def test_ends_with_md(self) -> None:
        fm = _fm()
        assert build_filename(fm).endswith(".md")
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_write_filename.py -v
```

Expected: FAIL.

- [ ] **Step 3: Append to `src/yt2md/stages/write.py`**

```python
from yt2md.models import Frontmatter


def build_filename(fm: Frontmatter) -> str:
    """Build the deterministic output filename for a structured doc.

    Format: {published-date}__{channel-slug}__{title-slug}.md
    On collision (handled by write()), the video_id is appended as a suffix.
    """
    return f"{fm.published.isoformat()}__{slugify(fm.channel)}__{slugify(fm.title)}.md"
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_write_filename.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/write.py tests/unit/test_write_filename.py
git commit -m "$(cat <<'EOF'
feat(write): add build_filename() from frontmatter

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task E.3: write() with atomic temp+rename + collision handling

**Files:**
- Create: `tests/unit/test_write_write.py`
- Modify: `src/yt2md/stages/write.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_write_write.py`:

```python
"""Tests for the write() function: atomic write + collision handling."""

from datetime import date
from pathlib import Path

import pytest

from yt2md.models import CURRENT_SCHEMA_VERSION, Frontmatter, StructuredDoc, Takeaway
from yt2md.stages.write import write


def _doc(title: str = "T", channel: str = "C", video_id: str = "vid") -> StructuredDoc:
    fm = Frontmatter(
        title=title,
        channel=channel,
        url=f"https://www.youtube.com/watch?v={video_id}",
        video_id=video_id,
        published=date(2024, 3, 15),
        duration_seconds=10,
        captured_at=date(2026, 5, 23),
        schema_version=CURRENT_SCHEMA_VERSION,
        genre="podcast",
        speakers=[],
        topics=[],
        people_mentioned=[],
        works_mentioned=[],
    )
    return StructuredDoc(
        frontmatter=fm,
        tldr="t",
        takeaways=[Takeaway(text="x", timestamp_s=0.0)],
        concepts=[],
        references=[],
        quotes=[],
        sections=[],
        open_questions=[],
        speaker_name_map={},
    )


class TestWriteHappyPath:
    def test_writes_to_correct_filename(self, tmp_path: Path) -> None:
        d = _doc(title="Dopamine", channel="Huberman Lab")
        path = write(markdown="hello", doc=d, output_dir=tmp_path)
        assert path == tmp_path / "2024-03-15__huberman-lab__dopamine.md"
        assert path.read_text(encoding="utf-8") == "hello"

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        d = _doc()
        out_dir = tmp_path / "nested" / "out"
        path = write(markdown="x", doc=d, output_dir=out_dir)
        assert path.parent == out_dir
        assert path.exists()

    def test_no_tmp_left_behind(self, tmp_path: Path) -> None:
        d = _doc()
        write(markdown="x", doc=d, output_dir=tmp_path)
        leftover = list(tmp_path.glob("*.tmp"))
        assert leftover == []


class TestCollision:
    def test_collision_with_different_video_id_appends_suffix(self, tmp_path: Path) -> None:
        # Pre-create a file with a different video_id's frontmatter
        existing = tmp_path / "2024-03-15__huberman-lab__dopamine.md"
        existing.write_text("---\nvideo_id: OTHER\n---\n", encoding="utf-8")

        d = _doc(title="Dopamine", channel="Huberman Lab", video_id="MINE")
        path = write(markdown="hello", doc=d, output_dir=tmp_path)
        assert path == tmp_path / "2024-03-15__huberman-lab__dopamine__MINE.md"
        assert path.read_text(encoding="utf-8") == "hello"
        # Existing file untouched
        assert existing.read_text(encoding="utf-8") == "---\nvideo_id: OTHER\n---\n"

    def test_collision_with_same_video_id_overwrites(self, tmp_path: Path) -> None:
        # Same video_id → overwrite (idempotency check would normally short-circuit
        # before reaching write(); this test exercises the lower-level behavior).
        existing = tmp_path / "2024-03-15__huberman-lab__dopamine.md"
        existing.write_text("---\nvideo_id: vid\n---\nOLD", encoding="utf-8")

        d = _doc(title="Dopamine", channel="Huberman Lab", video_id="vid")
        path = write(markdown="NEW", doc=d, output_dir=tmp_path)
        assert path == existing
        assert path.read_text(encoding="utf-8") == "NEW"
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_write_write.py -v
```

Expected: FAIL — `write` does not exist.

- [ ] **Step 3: Append to `src/yt2md/stages/write.py`**

```python
import re
from pathlib import Path

from yt2md.models import StructuredDoc


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
        # Collision with a different video → append suffix.
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
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_write_write.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/stages/write.py tests/unit/test_write_write.py
git commit -m "$(cat <<'EOF'
feat(write): write() with atomic temp+rename and video_id collision suffix

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 final check

### Task F.1: Run the full Phase 2 suite + coverage gate

- [ ] **Step 1: Run all unit tests with coverage**

```bash
uv run pytest tests/unit --cov=src/yt2md --cov-report=term-missing --cov-fail-under=85
```

Expected: all tests pass; coverage ≥85%.

- [ ] **Step 2: Run pre-commit on all files**

```bash
uv run pre-commit run --all-files
```

Expected: all hooks pass.

- [ ] **Step 3: Verify the golden render is still byte-exact**

```bash
uv run pytest tests/unit/test_render_golden.py -v
```

Expected: PASS. (If this fails, a template change broke the golden. Regenerate via the Step 2 script in Task D.6 and re-commit.)

- [ ] **Step 4: Mark Phase 2 complete in the index**

Edit `docs/superpowers/plans/2026-05-23-yt2llm-index.md`:

```markdown
- [x] Phase 2 — Deterministic stages
```

Commit:

```bash
git add docs/superpowers/plans/2026-05-23-yt2llm-index.md
git commit -m "$(cat <<'EOF'
docs(plan): mark Phase 2 complete

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## What Phase 2 produced

- `src/yt2md/vocab_hint.py` — `VocabularyHints` dataclass, `extract_hints()`, `format_for_openai()`, `format_for_whisper()`, `VOCAB_HINT_VERSION`, token-budget truncation
- `src/yt2md/stages/clean.py` — `clean()` with `HARD_FILLERS` removal, 95% speaker collapse, 1% noise drop, `CLEANER_VERSION`
- `src/yt2md/stages/render.py` — `render()` producing the full Tier 3 markdown via Jinja2
- `src/yt2md/templates/document.md.j2` — the canonical document template
- `src/yt2md/stages/write.py` — `slugify()`, `build_filename()`, `write()` with atomic temp+rename and collision handling
- Test fixtures (`short_solo`, `multi_speaker`, `huberman_sample`, `sample_doc`) reusable by Phase 3 and 4 tests
- Property tests asserting token budget and timestamp-preservation invariants
- A byte-exact golden test for `render()`

**Still missing for MVP:** download, compress, transcribe (both backends), structure, chunking, pipeline orchestrator, CLI. Those are Phases 3 and 4.

---

## Next: Phase 3

Open `docs/superpowers/plans/2026-05-23-yt2llm-phase-3-api-stages.md` once it's written.
