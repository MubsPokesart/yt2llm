# yt2llm Phase 1: Bootstrap + Foundations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-05-23-yt2llm-design.md`
**Index:** `docs/superpowers/plans/2026-05-23-yt2llm-index.md`

**Goal:** Stand up the project skeleton (pyproject, lint/type/test tooling, CI, pre-commit) and implement the five infrastructure modules (`errors`, `models`, `config`, `costs`, `cache`) with full TDD coverage.

**Architecture:** No runtime behavior in this phase — pure data contracts (Pydantic), pure utility (cache helper), and tooling. Everything testable without network or filesystem beyond `tmp_path`.

**Tech Stack:** Python 3.11+, uv, Ruff, mypy, pre-commit, pytest, hypothesis, Pydantic v2, pydantic-settings.

**Definition of done:** `lint` + `typecheck` + `cover` all pass. CI runs the same on push. All Phase 1 tasks checked off.

---

## Non-negotiable discipline (recap from index)

- **TDD:** failing test first, run-to-confirm-fails, minimum impl, run-to-confirm-passes, lint+typecheck, commit. No exceptions.
- **Lint+type gate:** every commit must pass `lint`, `typecheck`, `pytest -q`.
- **400 LOC per src/ module ceiling.**
- **No abstractions until 3 concrete uses.**
- **No code for hypothetical futures.**
- **Never `--no-verify`.**

Shorthand: `lint` = `ruff check src/ tests/ && ruff format --check src/ tests/`. `typecheck` = `mypy --strict src/`. `test <path>` = `pytest <path> -v`. `testall` = `pytest tests/unit tests/integration -q`. `cover` = `pytest tests/unit tests/integration --cov=src/yt2md --cov-report=term-missing --cov-fail-under=85`.

---

## Section A: Bootstrap (no test code yet — infra setup)

These tasks set up the project skeleton. They have no TDD cycle because they don't add behavior — they configure tools. Each task ends in a commit so the tooling is bisectable.

---

### Task 1.1: Initialize uv project and Python pin

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `README.md`
- Modify: `.gitignore` (create if missing)

- [ ] **Step 1: Initialize uv project**

```bash
uv init --package --name yt2llm --python 3.11
```

Expected: creates `pyproject.toml`, `src/yt2llm/`, `.python-version`.

- [ ] **Step 2: Rename package directory to match spec**

The spec uses `src/yt2md/` as the package directory (the CLI is `yt2md`, the project is `yt2llm`). Rename.

```bash
mv src/yt2llm src/yt2md
```

- [ ] **Step 3: Replace `pyproject.toml` with the canonical version**

Overwrite `pyproject.toml` with:

```toml
[project]
name = "yt2llm"
version = "0.0.1"
description = "Turn YouTube videos into structured markdown for an LLM knowledge graph"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "typer>=0.12",
    "rich>=13.7",
    "structlog>=24.1",
    "tenacity>=8.3",
    "jinja2>=3.1",
    "tiktoken>=0.7",
    "yt-dlp>=2024.5.27",
    "openai>=1.30",
    "google-genai>=0.3",
]

[project.optional-dependencies]
local = ["faster-whisper>=1.0"]

[project.scripts]
yt2md = "yt2md.cli:app"

[dependency-groups]
dev = [
    "pytest>=8.2",
    "pytest-cov>=5.0",
    "hypothesis>=6.100",
    "mypy>=1.10",
    "ruff>=0.5",
    "pre-commit>=3.7",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/yt2md"]

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
"tests/**" = ["S101", "PLR2004", "ANN", "PLR0915"]
"src/yt2md/prompts/**" = ["E501"]
"src/yt2md/templates/**" = ["E501"]

[tool.mypy]
strict = true
python_version = "3.11"
plugins = ["pydantic.mypy"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "live: tests that hit real external APIs (require keys)",
]
addopts = "-ra --strict-markers"

[tool.coverage.run]
source = ["src/yt2md"]
omit = ["src/yt2md/__main__.py"]

[tool.coverage.report]
exclude_also = [
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
    "@abstractmethod",
]
```

- [ ] **Step 4: Write README.md placeholder**

```markdown
# yt2llm

Turn YouTube videos into structured markdown for an LLM knowledge graph.

## Status

Under active development. See `docs/superpowers/specs/2026-05-23-yt2llm-design.md` for the design.

## Install

```bash
uv sync
```

For local-whisper fallback:

```bash
uv sync --extra local
```
```

- [ ] **Step 5: Create `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.eggs/

# Virtual environments
.venv/
venv/

# Testing
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
coverage.xml

# yt2llm artifacts
cache/
output/
runs.log

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
```

- [ ] **Step 6: Install dependencies**

```bash
uv sync --extra local
```

Expected: lockfile generated, venv created, all deps installed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .python-version .gitignore README.md uv.lock src/yt2md/
git commit -m "$(cat <<'EOF'
chore(bootstrap): initialize uv project with full tooling pyproject

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.2: Add pre-commit config + install hooks

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Write `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0      # bump to current latest at commit time
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0     # bump to current latest at commit time
    hooks:
      - id: mypy
        args: [--strict, src/]
        additional_dependencies:
          - pydantic>=2.7
          - pydantic-settings>=2.3
  - repo: local
    hooks:
      - id: max-file-lines
        name: max file lines (400)
        entry: python -c "import sys; [sys.exit(f'{f}: too long ({sum(1 for _ in open(f, encoding=\"utf-8\"))} > 400)') for f in sys.argv[1:] if sum(1 for _ in open(f, encoding='utf-8')) > 400]"
        language: system
        files: ^src/.*\.py$
      - id: pytest-fast
        name: pytest unit tests (fast, no network)
        entry: uv run pytest tests/unit -q
        language: system
        pass_filenames: false
        stages: [pre-commit]
```

If pinning to specific rev versions requires the current latest, fetch them with:

```bash
uv run pre-commit autoupdate
```

This rewrites the `rev:` fields to the latest stable releases. Inspect the diff before committing.

- [ ] **Step 2: Install the hooks**

```bash
uv run pre-commit install
```

Expected: `pre-commit installed at .git/hooks/pre-commit`.

- [ ] **Step 3: Run hooks against all current files**

```bash
uv run pre-commit run --all-files
```

Expected: all hooks pass (no source files yet beyond stubs).

- [ ] **Step 4: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "$(cat <<'EOF'
chore(tooling): add pre-commit with ruff, mypy, 400-LOC ceiling, fast pytest

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.3: Set up src package and test scaffolding

**Files:**
- Create: `src/yt2md/__init__.py`
- Create: `src/yt2md/__main__.py`
- Create: `src/yt2md/py.typed`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/live/__init__.py`
- Create: `tests/fixtures/.gitkeep`

- [ ] **Step 1: Write `src/yt2md/__init__.py`**

```python
"""yt2llm — YouTube videos to structured markdown for an LLM knowledge graph."""

__version__ = "0.0.1"
```

- [ ] **Step 2: Write `src/yt2md/__main__.py`**

```python
"""Allow `python -m yt2md` invocation."""

from yt2md.cli import app

if __name__ == "__main__":
    app()
```

(`cli.py` doesn't exist yet — that's Phase 4. mypy will currently flag this; we'll exclude `__main__.py` from coverage and accept the mypy error until Phase 4. For now, replace the body with a stub:)

Replace contents with:

```python
"""Allow `python -m yt2md` invocation. CLI implementation lands in Phase 4."""

def main() -> None:
    """Placeholder entry point until cli.py exists."""
    raise SystemExit("yt2md CLI not yet implemented")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create marker files**

```bash
touch src/yt2md/py.typed
mkdir -p tests/unit tests/integration tests/live tests/fixtures
touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py tests/live/__init__.py tests/fixtures/.gitkeep
```

- [ ] **Step 4: Write `tests/conftest.py`**

```python
"""Shared pytest configuration and fixtures."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the tests/fixtures directory."""
    return FIXTURES_DIR
```

- [ ] **Step 5: Run lint + typecheck**

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy --strict src/
```

Expected: all pass. (Ruff may want to auto-format; run `uv run ruff format src/ tests/` if needed.)

- [ ] **Step 6: Commit**

```bash
git add src/yt2md/ tests/
git commit -m "$(cat <<'EOF'
chore(scaffold): add src/yt2md package structure and tests/ layout

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 1.4: Add CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write CI workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync --all-extras --dev

      - name: Lint
        run: |
          uv run ruff check src/ tests/ --output-format=github
          uv run ruff format --check src/ tests/

      - name: Type check
        run: uv run mypy --strict src/

      - name: Test
        run: uv run pytest tests/unit tests/integration --cov=src/yt2md --cov-report=term-missing --cov-fail-under=85
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
ci: add GitHub Actions workflow with lint, typecheck, coverage gate

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Note: coverage gate will fail on the next push because no source files have tests yet. That's fine — the next tasks will add tests that satisfy it. CI is intentionally added before tests to confirm the gate is real.)

---

### Task 1.5: First sanity test — version is exposed

**Files:**
- Create: `tests/unit/test_version.py`

This is the "is the test infrastructure working?" smoke test. Tiny but real TDD: fails first, then passes.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_version.py`:

```python
"""Smoke test: package version is importable."""

import yt2md


def test_version_is_a_string() -> None:
    assert isinstance(yt2md.__version__, str)


def test_version_is_nonempty() -> None:
    assert yt2md.__version__
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/unit/test_version.py -v
```

Expected: PASS (because `__init__.py` already exports `__version__`). If it FAILS, fix the import in `__init__.py`.

This task is intentionally green-on-first-run because we already wrote `__version__ = "0.0.1"`. It's the smoke test that pytest + import paths work. Subsequent tasks will follow strict red-first.

- [ ] **Step 3: Run lint + typecheck**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_version.py
git commit -m "$(cat <<'EOF'
test(version): smoke test that yt2md.__version__ is exposed

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section B: errors.py (typed exception hierarchy)

---

### Task 2.1: Exception hierarchy

**Files:**
- Create: `tests/unit/test_errors.py`
- Create: `src/yt2md/errors.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_errors.py`:

```python
"""Tests for the typed exception hierarchy in src/yt2md/errors.py."""

import pytest

from yt2md.errors import (
    AudioTooLargeError,
    ConfigError,
    DownloadError,
    InvalidStructuredOutputError,
    LivestreamNotEndedError,
    NoAudioStreamError,
    StructuringError,
    TranscriptionError,
    VideoUnavailableError,
    WriteError,
    YT2MDError,
)


class TestRootException:
    def test_yt2md_error_is_exception(self) -> None:
        assert issubclass(YT2MDError, Exception)

    def test_yt2md_error_carries_message(self) -> None:
        err = YT2MDError("boom")
        assert str(err) == "boom"


class TestConfigBranch:
    def test_config_error_inherits_root(self) -> None:
        assert issubclass(ConfigError, YT2MDError)


class TestDownloadBranch:
    def test_download_error_inherits_root(self) -> None:
        assert issubclass(DownloadError, YT2MDError)

    @pytest.mark.parametrize(
        "subclass",
        [VideoUnavailableError, LivestreamNotEndedError, NoAudioStreamError],
    )
    def test_subclass_inherits_download(self, subclass: type[Exception]) -> None:
        assert issubclass(subclass, DownloadError)


class TestTranscriptionBranch:
    def test_transcription_error_inherits_root(self) -> None:
        assert issubclass(TranscriptionError, YT2MDError)

    def test_audio_too_large_inherits_transcription(self) -> None:
        assert issubclass(AudioTooLargeError, TranscriptionError)


class TestStructuringBranch:
    def test_structuring_error_inherits_root(self) -> None:
        assert issubclass(StructuringError, YT2MDError)

    def test_invalid_structured_output_inherits_structuring(self) -> None:
        assert issubclass(InvalidStructuredOutputError, StructuringError)


class TestWriteBranch:
    def test_write_error_inherits_root(self) -> None:
        assert issubclass(WriteError, YT2MDError)


class TestRaiseAndCatch:
    def test_specific_caught_as_root(self) -> None:
        with pytest.raises(YT2MDError):
            raise VideoUnavailableError("private video")

    def test_specific_not_caught_as_sibling(self) -> None:
        with pytest.raises(VideoUnavailableError):
            try:
                raise VideoUnavailableError("private")
            except StructuringError:  # pragma: no cover  -- impossible
                pytest.fail("VideoUnavailableError must not match StructuringError")
            except VideoUnavailableError:
                raise
```

- [ ] **Step 2: Run the test — confirm it fails**

```bash
uv run pytest tests/unit/test_errors.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'yt2md.errors'`.

- [ ] **Step 3: Write the implementation**

`src/yt2md/errors.py`:

```python
"""Typed exception hierarchy for yt2llm.

Every exception raised by stages or pipeline orchestration is a subclass of YT2MDError.
CLI exit codes are mapped at the catch site in cli.py.
"""


class YT2MDError(Exception):
    """Root of the yt2llm exception hierarchy. Catch this to handle any pipeline error."""


class ConfigError(YT2MDError):
    """Configuration or environment problem. Exit code 3."""


class DownloadError(YT2MDError):
    """Failure in the download stage. Exit code 1 unless a more specific subclass."""


class VideoUnavailableError(DownloadError):
    """Video cannot be accessed: private, removed, age-restricted, members-only, geoblocked. Exit code 2."""


class LivestreamNotEndedError(DownloadError):
    """The URL points to a livestream that is still active. Exit code 2."""


class NoAudioStreamError(DownloadError):
    """The video has no audio track. Exit code 2."""


class TranscriptionError(YT2MDError):
    """Failure in the transcribe stage. Exit code 1."""


class AudioTooLargeError(TranscriptionError):
    """Audio exceeds the backend's per-request limit even after chunking. Exit code 1."""


class StructuringError(YT2MDError):
    """Failure in the structure stage. Exit code 1."""


class InvalidStructuredOutputError(StructuringError):
    """Gemini output failed Pydantic or semantic validation after retry. Exit code 1."""


class WriteError(YT2MDError):
    """Failure writing the final markdown to disk. Exit code 1."""
```

- [ ] **Step 4: Run the test — confirm it passes**

```bash
uv run pytest tests/unit/test_errors.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Run lint + typecheck**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/yt2md/errors.py tests/unit/test_errors.py
git commit -m "$(cat <<'EOF'
feat(errors): add typed exception hierarchy with full coverage

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section C: models.py (Pydantic data contracts)

Five tasks, one per logical cluster of models. Each task is full TDD: failing test, implementation, lint, commit.

---

### Task 3.1: Word, Segment, Transcript

**Files:**
- Create: `tests/unit/test_models_transcript.py`
- Create: `src/yt2md/models.py` (start it)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_models_transcript.py`:

```python
"""Tests for Word, Segment, and Transcript models."""

import pytest
from pydantic import ValidationError

from yt2md.models import Segment, Transcript, Word


class TestWord:
    def test_minimal_word(self) -> None:
        w = Word(text="hello", start=0.0, end=0.5, speaker=None)
        assert w.text == "hello"
        assert w.start == 0.0
        assert w.end == 0.5
        assert w.speaker is None

    def test_word_with_speaker(self) -> None:
        w = Word(text="hi", start=1.0, end=1.2, speaker="SPEAKER_00")
        assert w.speaker == "SPEAKER_00"

    def test_end_before_start_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Word(text="bad", start=1.0, end=0.5, speaker=None)

    def test_negative_start_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Word(text="bad", start=-0.1, end=0.5, speaker=None)


class TestSegment:
    def test_segment_with_words(self) -> None:
        words = [
            Word(text="hello", start=0.0, end=0.5, speaker="S0"),
            Word(text="world", start=0.6, end=1.0, speaker="S0"),
        ]
        s = Segment(
            start=0.0,
            end=1.0,
            text="hello world",
            speaker="S0",
            words=words,
        )
        assert len(s.words) == 2
        assert s.text == "hello world"


class TestTranscript:
    def test_transcript_minimal(self) -> None:
        t = Transcript(
            language="en",
            duration_s=1.0,
            backend="openai_transcribe",
            model_id="gpt-4o-transcribe",
            chunked=False,
            segments=[
                Segment(
                    start=0.0,
                    end=1.0,
                    text="hi",
                    speaker=None,
                    words=[Word(text="hi", start=0.0, end=1.0, speaker=None)],
                ),
            ],
            speakers=[],
        )
        assert t.backend == "openai_transcribe"
        assert t.chunked is False
        assert len(t.segments) == 1

    def test_invalid_backend_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Transcript(
                language="en",
                duration_s=1.0,
                backend="not_a_backend",  # type: ignore[arg-type]
                model_id="x",
                chunked=False,
                segments=[],
                speakers=[],
            )

    def test_round_trip_json(self) -> None:
        t = Transcript(
            language="en",
            duration_s=2.5,
            backend="local_whisper",
            model_id="faster-whisper-medium",
            chunked=True,
            segments=[],
            speakers=["Alice"],
        )
        as_json = t.model_dump_json()
        back = Transcript.model_validate_json(as_json)
        assert back == t
```

- [ ] **Step 2: Run the test — confirm it fails**

```bash
uv run pytest tests/unit/test_models_transcript.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'yt2md.models'`.

- [ ] **Step 3: Write the implementation**

`src/yt2md/models.py`:

```python
"""Pydantic v2 data contracts for yt2llm.

Every artifact passed between stages is one of these types. The schema is the contract;
deviations cause loud, typed errors at the stage boundary.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Word(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    speaker: str | None

    @model_validator(mode="after")
    def end_after_start(self) -> Word:
        if self.end < self.start:
            msg = f"Word.end ({self.end}) must be >= Word.start ({self.start})"
            raise ValueError(msg)
        return self


class Segment(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str
    speaker: str | None
    words: list[Word]


Backend = Literal["openai_transcribe", "local_whisper"]


class Transcript(BaseModel):
    model_config = ConfigDict(frozen=True)

    language: str
    duration_s: float = Field(ge=0.0)
    backend: Backend
    model_id: str
    chunked: bool
    segments: list[Segment]
    speakers: list[str]
```

- [ ] **Step 4: Run the test — confirm it passes**

```bash
uv run pytest tests/unit/test_models_transcript.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Run lint + typecheck**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/yt2md/models.py tests/unit/test_models_transcript.py
git commit -m "$(cat <<'EOF'
feat(models): add Word, Segment, Transcript with strict validation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3.2: Chapter and VideoMetadata

**Files:**
- Create: `tests/unit/test_models_metadata.py`
- Modify: `src/yt2md/models.py` (append)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_models_metadata.py`:

```python
"""Tests for Chapter and VideoMetadata models."""

from datetime import date

import pytest
from pydantic import ValidationError

from yt2md.models import Chapter, VideoMetadata


class TestChapter:
    def test_chapter(self) -> None:
        c = Chapter(title="Intro", start_s=0.0, end_s=60.0)
        assert c.title == "Intro"

    def test_end_before_start_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Chapter(title="bad", start_s=10.0, end_s=5.0)


class TestVideoMetadata:
    def test_minimal(self) -> None:
        m = VideoMetadata(
            video_id="abc123",
            url="https://www.youtube.com/watch?v=abc123",
            title="Test Video",
            channel="Test Channel",
            channel_id="UCxxxx",
            published_date=date(2024, 3, 15),
            duration_s=5025.0,
            description="A test video",
            chapters=[],
            tags=[],
            language=None,
        )
        assert m.video_id == "abc123"
        assert m.published_date == date(2024, 3, 15)
        assert m.chapters == []

    def test_with_chapters(self) -> None:
        m = VideoMetadata(
            video_id="abc",
            url="https://www.youtube.com/watch?v=abc",
            title="T",
            channel="C",
            channel_id="UC1",
            published_date=date(2025, 1, 1),
            duration_s=120.0,
            description="",
            chapters=[
                Chapter(title="Intro", start_s=0.0, end_s=30.0),
                Chapter(title="Main", start_s=30.0, end_s=120.0),
            ],
            tags=["science", "neuroscience"],
            language="en",
        )
        assert len(m.chapters) == 2
        assert m.tags == ["science", "neuroscience"]
```

- [ ] **Step 2: Run the test — confirm it fails**

```bash
uv run pytest tests/unit/test_models_metadata.py -v
```

Expected: FAIL with `ImportError: cannot import name 'Chapter' from 'yt2md.models'`.

- [ ] **Step 3: Append to `src/yt2md/models.py`**

Add at the bottom of the file (after `Transcript`):

```python
class Chapter(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)

    @model_validator(mode="after")
    def end_after_start(self) -> Chapter:
        if self.end_s < self.start_s:
            msg = f"Chapter.end_s ({self.end_s}) must be >= Chapter.start_s ({self.start_s})"
            raise ValueError(msg)
        return self


class VideoMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    url: str
    title: str
    channel: str
    channel_id: str
    published_date: date
    duration_s: float = Field(ge=0.0)
    description: str
    chapters: list[Chapter]
    tags: list[str]
    language: str | None
```

And add the import at the top (after `Literal`):

```python
from datetime import date
```

- [ ] **Step 4: Run the test — confirm it passes**

```bash
uv run pytest tests/unit/test_models_metadata.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Run lint + typecheck**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/yt2md/models.py tests/unit/test_models_metadata.py
git commit -m "$(cat <<'EOF'
feat(models): add Chapter and VideoMetadata

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3.3: Structured doc item models (Reference, Takeaway, Concept, Quote, DetailedSection)

**Files:**
- Create: `tests/unit/test_models_items.py`
- Modify: `src/yt2md/models.py` (append)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_models_items.py`:

```python
"""Tests for structured-doc item models: Reference, Takeaway, Concept, Quote, DetailedSection."""

import pytest
from pydantic import ValidationError

from yt2md.models import Concept, DetailedSection, Quote, Reference, Takeaway


class TestTakeaway:
    def test_takeaway(self) -> None:
        t = Takeaway(text="Dopamine signals anticipation.", timestamp_s=252.0)
        assert t.timestamp_s == 252.0

    def test_negative_timestamp_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Takeaway(text="x", timestamp_s=-1.0)


class TestConcept:
    def test_concept(self) -> None:
        c = Concept(name="Reward Prediction Error", definition="Gap between expected and actual reward.", timestamp_s=510.0)
        assert c.name == "Reward Prediction Error"


class TestReference:
    @pytest.mark.parametrize(
        "kind",
        ["book", "paper", "person", "tool", "video", "other"],
    )
    def test_reference_kinds(self, kind: str) -> None:
        r = Reference(kind=kind, name="X", context="ctx", timestamp_s=0.0)  # type: ignore[arg-type]
        assert r.kind == kind

    def test_invalid_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Reference(kind="movie", name="X", context="c", timestamp_s=0.0)  # type: ignore[arg-type]


class TestQuote:
    def test_quote(self) -> None:
        q = Quote(text="Pursuit, not pleasure.", speaker="Andrew Huberman", timestamp_s=754.0)
        assert q.speaker == "Andrew Huberman"


class TestDetailedSection:
    def test_section(self) -> None:
        s = DetailedSection(heading="What dopamine actually does", body="Multi paragraph.", timestamp_s=0.0)
        assert s.heading.startswith("What")
```

- [ ] **Step 2: Run the test — confirm it fails**

```bash
uv run pytest tests/unit/test_models_items.py -v
```

Expected: FAIL with import errors.

- [ ] **Step 3: Append to `src/yt2md/models.py`**

Add at the bottom:

```python
ReferenceKind = Literal["book", "paper", "person", "tool", "video", "other"]


class Reference(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: ReferenceKind
    name: str
    context: str
    timestamp_s: float = Field(ge=0.0)


class Takeaway(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    timestamp_s: float = Field(ge=0.0)


class Concept(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    definition: str
    timestamp_s: float = Field(ge=0.0)


class Quote(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    speaker: str
    timestamp_s: float = Field(ge=0.0)


class DetailedSection(BaseModel):
    model_config = ConfigDict(frozen=True)

    heading: str
    body: str
    timestamp_s: float = Field(ge=0.0)
```

- [ ] **Step 4: Run the test — confirm it passes**

```bash
uv run pytest tests/unit/test_models_items.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Run lint + typecheck**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/yt2md/models.py tests/unit/test_models_items.py
git commit -m "$(cat <<'EOF'
feat(models): add Reference, Takeaway, Concept, Quote, DetailedSection

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3.4: Frontmatter

**Files:**
- Create: `tests/unit/test_models_frontmatter.py`
- Modify: `src/yt2md/models.py` (append)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_models_frontmatter.py`:

```python
"""Tests for the Frontmatter model."""

from datetime import date

import pytest
from pydantic import ValidationError

from yt2md.models import Frontmatter


class TestFrontmatter:
    def test_full(self) -> None:
        fm = Frontmatter(
            title="Dopamine, Motivation & Drive",
            channel="Huberman Lab",
            url="https://www.youtube.com/watch?v=abc",
            video_id="abc",
            published=date(2024, 3, 15),
            duration_seconds=5025,
            captured_at=date(2026, 5, 23),
            schema_version=1,
            genre="podcast",
            speakers=["Andrew Huberman"],
            topics=["dopamine", "motivation"],
            people_mentioned=["Robert Sapolsky"],
            works_mentioned=["The Molecule of More"],
        )
        assert fm.schema_version == 1
        assert fm.genre == "podcast"

    @pytest.mark.parametrize(
        "genre",
        ["podcast", "lecture", "tutorial", "talk", "interview", "other"],
    )
    def test_genre_enum(self, genre: str) -> None:
        fm = Frontmatter(
            title="t",
            channel="c",
            url="u",
            video_id="v",
            published=date(2025, 1, 1),
            duration_seconds=1,
            captured_at=date(2025, 1, 1),
            schema_version=1,
            genre=genre,  # type: ignore[arg-type]
            speakers=[],
            topics=[],
            people_mentioned=[],
            works_mentioned=[],
        )
        assert fm.genre == genre

    def test_invalid_genre_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Frontmatter(
                title="t",
                channel="c",
                url="u",
                video_id="v",
                published=date(2025, 1, 1),
                duration_seconds=1,
                captured_at=date(2025, 1, 1),
                schema_version=1,
                genre="movie",  # type: ignore[arg-type]
                speakers=[],
                topics=[],
                people_mentioned=[],
                works_mentioned=[],
            )
```

- [ ] **Step 2: Run the test — confirm it fails**

```bash
uv run pytest tests/unit/test_models_frontmatter.py -v
```

Expected: FAIL.

- [ ] **Step 3: Append to `src/yt2md/models.py`**

```python
Genre = Literal["podcast", "lecture", "tutorial", "talk", "interview", "other"]


class Frontmatter(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    channel: str
    url: str
    video_id: str
    published: date
    duration_seconds: int = Field(ge=0)
    captured_at: date
    schema_version: int = Field(ge=1)
    genre: Genre
    speakers: list[str]
    topics: list[str]
    people_mentioned: list[str]
    works_mentioned: list[str]
```

- [ ] **Step 4: Run the test — confirm it passes**

```bash
uv run pytest tests/unit/test_models_frontmatter.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/models.py tests/unit/test_models_frontmatter.py
git commit -m "$(cat <<'EOF'
feat(models): add Frontmatter with genre enum and schema_version field

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3.5: StructuredDoc + CURRENT_SCHEMA_VERSION

**Files:**
- Create: `tests/unit/test_models_structured_doc.py`
- Modify: `src/yt2md/models.py` (append)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_models_structured_doc.py`:

```python
"""Tests for the StructuredDoc root model + CURRENT_SCHEMA_VERSION constant."""

from datetime import date

from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    Concept,
    DetailedSection,
    Frontmatter,
    Quote,
    Reference,
    StructuredDoc,
    Takeaway,
)


def _make_frontmatter() -> Frontmatter:
    return Frontmatter(
        title="t",
        channel="c",
        url="u",
        video_id="v",
        published=date(2025, 1, 1),
        duration_seconds=10,
        captured_at=date(2025, 1, 1),
        schema_version=CURRENT_SCHEMA_VERSION,
        genre="podcast",
        speakers=["A"],
        topics=[],
        people_mentioned=[],
        works_mentioned=[],
    )


class TestStructuredDoc:
    def test_minimal(self) -> None:
        doc = StructuredDoc(
            frontmatter=_make_frontmatter(),
            tldr="Short.",
            takeaways=[Takeaway(text="x", timestamp_s=0.0)],
            concepts=[],
            references=[],
            quotes=[],
            sections=[],
            open_questions=[],
            speaker_name_map={"SPEAKER_00": "A"},
        )
        assert doc.tldr == "Short."

    def test_round_trip_json(self) -> None:
        doc = StructuredDoc(
            frontmatter=_make_frontmatter(),
            tldr="Hello.",
            takeaways=[Takeaway(text="x", timestamp_s=0.0)],
            concepts=[Concept(name="N", definition="D", timestamp_s=1.0)],
            references=[Reference(kind="book", name="B", context="c", timestamp_s=2.0)],
            quotes=[Quote(text="q", speaker="A", timestamp_s=3.0)],
            sections=[DetailedSection(heading="H", body="B", timestamp_s=4.0)],
            open_questions=["?"],
            speaker_name_map={"SPEAKER_00": "A"},
        )
        back = StructuredDoc.model_validate_json(doc.model_dump_json())
        assert back == doc


class TestSchemaVersion:
    def test_current_version_is_positive_int(self) -> None:
        assert isinstance(CURRENT_SCHEMA_VERSION, int)
        assert CURRENT_SCHEMA_VERSION >= 1
```

- [ ] **Step 2: Run the test — confirm it fails**

```bash
uv run pytest tests/unit/test_models_structured_doc.py -v
```

Expected: FAIL on import of `StructuredDoc` or `CURRENT_SCHEMA_VERSION`.

- [ ] **Step 3: Append to `src/yt2md/models.py`**

```python
# Bump this when the StructuredDoc schema changes meaning (renamed fields,
# restructured sections, changed semantics). Adding an optional field does NOT bump.
# Bumping invalidates schema_version checks in existing output files and is
# required to make `yt2md regen --all` regenerate them.
CURRENT_SCHEMA_VERSION = 1


class StructuredDoc(BaseModel):
    model_config = ConfigDict(frozen=True)

    frontmatter: Frontmatter
    tldr: str
    takeaways: list[Takeaway]
    concepts: list[Concept]
    references: list[Reference]
    quotes: list[Quote]
    sections: list[DetailedSection]
    open_questions: list[str]
    speaker_name_map: dict[str, str]
```

- [ ] **Step 4: Run the test — confirm it passes**

```bash
uv run pytest tests/unit/test_models_structured_doc.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Run full test suite + lint + typecheck**

```bash
uv run pytest tests/unit -v
uv run ruff check src/ tests/
uv run mypy --strict src/
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/yt2md/models.py tests/unit/test_models_structured_doc.py
git commit -m "$(cat <<'EOF'
feat(models): add StructuredDoc root and CURRENT_SCHEMA_VERSION constant

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section D: config.py (pydantic-settings)

---

### Task 4.1: Config model with defaults

**Files:**
- Create: `tests/unit/test_config_defaults.py`
- Create: `src/yt2md/config.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_config_defaults.py`:

```python
"""Tests for Config defaults (no env, no TOML)."""

from pathlib import Path

import pytest

from yt2md.config import Config


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any YT2MD_* env vars so defaults are observable."""
    for key in list((__import__("os")).environ):
        if key.startswith("YT2MD_") or key in {"OPENAI_API_KEY", "GOOGLE_API_KEY"}:
            monkeypatch.delenv(key, raising=False)


class TestDefaults:
    def test_default_output_dir(self) -> None:
        cfg = Config(google_api_key="g")  # type: ignore[arg-type]
        assert cfg.output_dir == Path("./output")

    def test_default_cache_dir(self) -> None:
        cfg = Config(google_api_key="g")  # type: ignore[arg-type]
        assert cfg.cache_dir == Path("./cache")

    def test_default_audio_bitrate(self) -> None:
        cfg = Config(google_api_key="g")  # type: ignore[arg-type]
        assert cfg.audio_bitrate_kbps == 32

    def test_default_transcription_backend(self) -> None:
        cfg = Config(google_api_key="g")  # type: ignore[arg-type]
        assert cfg.transcription_backend == "auto"

    def test_default_force_and_no_cache_false(self) -> None:
        cfg = Config(google_api_key="g")  # type: ignore[arg-type]
        assert cfg.force is False
        assert cfg.no_cache is False

    def test_google_api_key_required(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Config()  # type: ignore[call-arg]

    def test_openai_api_key_optional(self) -> None:
        cfg = Config(google_api_key="g")  # type: ignore[arg-type]
        assert cfg.openai_api_key is None
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_config_defaults.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write `src/yt2md/config.py`**

```python
"""Configuration loading for yt2llm.

Precedence (12-factor): CLI flag > env var > TOML file > default.

env_prefix is YT2MD_. So YT2MD_OUTPUT_DIR sets output_dir.
TOML files searched (later wins): ~/.config/yt2md/config.toml, ./yt2md.toml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YT2MD_",
        env_file=".env",
        toml_file=[
            Path.home() / ".config" / "yt2md" / "config.toml",
            Path("yt2md.toml"),
        ],
        extra="ignore",
    )

    # API keys
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

    # CLI behavior flags
    force: bool = False
    no_cache: bool = False
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_config_defaults.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/config.py tests/unit/test_config_defaults.py
git commit -m "$(cat <<'EOF'
feat(config): add pydantic-settings Config with defaults

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4.2: Config precedence (env > defaults)

**Files:**
- Create: `tests/unit/test_config_precedence.py`

(No new implementation — this test exercises behavior pydantic-settings already provides. The test confirms the wiring is correct.)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_config_precedence.py`:

```python
"""Tests for Config precedence: env > TOML > defaults."""

from pathlib import Path

import pytest

from yt2md.config import Config


class TestEnvOverridesDefault:
    def test_output_dir_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT2MD_OUTPUT_DIR", "/tmp/yt2md-out")
        monkeypatch.setenv("YT2MD_GOOGLE_API_KEY", "g")
        cfg = Config()  # type: ignore[call-arg]
        assert cfg.output_dir == Path("/tmp/yt2md-out")

    def test_audio_bitrate_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT2MD_AUDIO_BITRATE_KBPS", "64")
        monkeypatch.setenv("YT2MD_GOOGLE_API_KEY", "g")
        cfg = Config()  # type: ignore[call-arg]
        assert cfg.audio_bitrate_kbps == 64

    def test_backend_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT2MD_TRANSCRIPTION_BACKEND", "local_whisper")
        monkeypatch.setenv("YT2MD_GOOGLE_API_KEY", "g")
        cfg = Config()  # type: ignore[call-arg]
        assert cfg.transcription_backend == "local_whisper"

    def test_invalid_backend_value_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pydantic import ValidationError

        monkeypatch.setenv("YT2MD_TRANSCRIPTION_BACKEND", "not_valid")
        monkeypatch.setenv("YT2MD_GOOGLE_API_KEY", "g")
        with pytest.raises(ValidationError):
            Config()  # type: ignore[call-arg]


class TestKwargOverridesEnv:
    def test_kwarg_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT2MD_OUTPUT_DIR", "/from/env")
        cfg = Config(google_api_key="g", output_dir=Path("/from/kwarg"))  # type: ignore[arg-type]
        assert cfg.output_dir == Path("/from/kwarg")


class TestSecretsNotLogged:
    def test_secret_str_repr_hidden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT2MD_GOOGLE_API_KEY", "supersecret")
        cfg = Config()  # type: ignore[call-arg]
        assert "supersecret" not in repr(cfg)
        assert cfg.google_api_key.get_secret_value() == "supersecret"
```

- [ ] **Step 2: Run — confirm passes (no new implementation needed)**

```bash
uv run pytest tests/unit/test_config_precedence.py -v
```

Expected: all 6 tests PASS. If any fail, the Config wiring is broken; fix the field name or env_prefix.

- [ ] **Step 3: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add tests/unit/test_config_precedence.py
git commit -m "$(cat <<'EOF'
test(config): cover env-over-default precedence and SecretStr opacity

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section E: costs.py (per-model cost calculators)

---

### Task 5.1: Transcription cost calculators

**Files:**
- Create: `tests/unit/test_costs_transcription.py`
- Create: `src/yt2md/costs.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_costs_transcription.py`:

```python
"""Tests for transcription cost calculators."""

import pytest

from yt2md.costs import local_whisper_cost, openai_transcribe_cost


class TestOpenAITranscribe:
    def test_one_minute(self) -> None:
        # $0.006 per minute as of mid-2025 (codified; bump on price changes)
        cost = openai_transcribe_cost(duration_s=60.0)
        assert cost == pytest.approx(0.006, rel=1e-3)

    def test_zero_duration(self) -> None:
        assert openai_transcribe_cost(duration_s=0.0) == 0.0

    def test_two_hours(self) -> None:
        cost = openai_transcribe_cost(duration_s=2 * 3600.0)
        assert cost == pytest.approx(0.72, rel=1e-3)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            openai_transcribe_cost(duration_s=-1.0)


class TestLocalWhisper:
    def test_is_zero(self) -> None:
        assert local_whisper_cost(duration_s=3600.0) == 0.0

    def test_zero_for_zero(self) -> None:
        assert local_whisper_cost(duration_s=0.0) == 0.0
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_costs_transcription.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write `src/yt2md/costs.py`**

```python
"""Per-model cost calculators. Single source of truth for pricing.

Rates are in USD and reflect public pricing as of mid-2025. When provider pricing
changes, bump the constants and add a regression test for the new value.
"""

OPENAI_TRANSCRIBE_USD_PER_MINUTE = 0.006

# Gemini 3 Flash (reasoning) pricing per million tokens
GEMINI_3_FLASH_INPUT_USD_PER_MTOK = 0.50
GEMINI_3_FLASH_OUTPUT_USD_PER_MTOK = 3.00


def openai_transcribe_cost(*, duration_s: float) -> float:
    """Cost in USD for transcribing `duration_s` seconds with gpt-4o-transcribe."""
    if duration_s < 0:
        msg = "duration_s must be non-negative"
        raise ValueError(msg)
    return (duration_s / 60.0) * OPENAI_TRANSCRIBE_USD_PER_MINUTE


def local_whisper_cost(*, duration_s: float) -> float:
    """Cost in USD for local-whisper transcription (always 0; CPU time is not billed here)."""
    if duration_s < 0:
        msg = "duration_s must be non-negative"
        raise ValueError(msg)
    return 0.0
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_costs_transcription.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/costs.py tests/unit/test_costs_transcription.py
git commit -m "$(cat <<'EOF'
feat(costs): add transcription cost calculators (openai + local)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5.2: Gemini cost calculator

**Files:**
- Create: `tests/unit/test_costs_gemini.py`
- Modify: `src/yt2md/costs.py` (append)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_costs_gemini.py`:

```python
"""Tests for Gemini cost calculator."""

import pytest

from yt2md.costs import gemini_flash_cost


class TestGeminiFlash:
    def test_one_million_input(self) -> None:
        # $0.50 per 1M input tokens
        cost = gemini_flash_cost(input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(0.50, rel=1e-3)

    def test_one_million_output(self) -> None:
        cost = gemini_flash_cost(input_tokens=0, output_tokens=1_000_000)
        assert cost == pytest.approx(3.00, rel=1e-3)

    def test_mixed(self) -> None:
        cost = gemini_flash_cost(input_tokens=25_000, output_tokens=10_000)
        # 25k * 0.50/1M = 0.0125; 10k * 3.00/1M = 0.030; total 0.0425
        assert cost == pytest.approx(0.0425, rel=1e-3)

    def test_zero(self) -> None:
        assert gemini_flash_cost(input_tokens=0, output_tokens=0) == 0.0

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            gemini_flash_cost(input_tokens=-1, output_tokens=0)
        with pytest.raises(ValueError, match="non-negative"):
            gemini_flash_cost(input_tokens=0, output_tokens=-1)
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_costs_gemini.py -v
```

Expected: FAIL.

- [ ] **Step 3: Append to `src/yt2md/costs.py`**

```python
def gemini_flash_cost(*, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD for a Gemini 3 Flash call with the given token counts."""
    if input_tokens < 0 or output_tokens < 0:
        msg = "token counts must be non-negative"
        raise ValueError(msg)
    in_cost = (input_tokens / 1_000_000.0) * GEMINI_3_FLASH_INPUT_USD_PER_MTOK
    out_cost = (output_tokens / 1_000_000.0) * GEMINI_3_FLASH_OUTPUT_USD_PER_MTOK
    return in_cost + out_cost
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_costs_gemini.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/costs.py tests/unit/test_costs_gemini.py
git commit -m "$(cat <<'EOF'
feat(costs): add Gemini 3 Flash cost calculator

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Section F: cache.py (artifact paths + cached helper)

---

### Task 6.1: ArtifactPaths — path resolution

**Files:**
- Create: `tests/unit/test_cache_paths.py`
- Create: `src/yt2md/cache.py`

- [ ] **Step 1: Write the failing test**

`tests/unit/test_cache_paths.py`:

```python
"""Tests for ArtifactPaths path resolution."""

from pathlib import Path

from yt2md.cache import ArtifactPaths


class TestArtifactPaths:
    def test_root(self, tmp_path: Path) -> None:
        paths = ArtifactPaths(cache_dir=tmp_path, video_id="abc123")
        assert paths.root == tmp_path / "abc123"

    def test_metadata(self, tmp_path: Path) -> None:
        paths = ArtifactPaths(cache_dir=tmp_path, video_id="abc")
        assert paths.metadata == tmp_path / "abc" / "metadata.json"

    def test_metadata_raw(self, tmp_path: Path) -> None:
        paths = ArtifactPaths(cache_dir=tmp_path, video_id="abc")
        assert paths.metadata_raw == tmp_path / "abc" / "metadata.raw.json"

    def test_audio_includes_compression_hash(self, tmp_path: Path) -> None:
        paths = ArtifactPaths(cache_dir=tmp_path, video_id="abc")
        p = paths.audio(compression_hash="deadbeef")
        assert p == tmp_path / "abc" / "audio-deadbeef.opus"

    def test_transcript_includes_input_hash(self, tmp_path: Path) -> None:
        paths = ArtifactPaths(cache_dir=tmp_path, video_id="abc")
        p = paths.transcript(input_hash="cafebabe")
        assert p == tmp_path / "abc" / "transcript-cafebabe.json"

    def test_transcript_raw(self, tmp_path: Path) -> None:
        paths = ArtifactPaths(cache_dir=tmp_path, video_id="abc")
        p = paths.transcript_raw(input_hash="cafebabe")
        assert p == tmp_path / "abc" / "transcript-cafebabe.raw.json"

    def test_cleaned(self, tmp_path: Path) -> None:
        paths = ArtifactPaths(cache_dir=tmp_path, video_id="abc")
        p = paths.cleaned(input_hash="aabb")
        assert p == tmp_path / "abc" / "cleaned-aabb.json"

    def test_structured(self, tmp_path: Path) -> None:
        paths = ArtifactPaths(cache_dir=tmp_path, video_id="abc")
        p = paths.structured(input_hash="ccdd")
        assert p == tmp_path / "abc" / "structured-ccdd.json"
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_cache_paths.py -v
```

Expected: FAIL.

- [ ] **Step 3: Write `src/yt2md/cache.py`**

```python
"""On-disk artifact cache for yt2llm.

Each stage's output lives at a path that includes a short hash of everything that
affects it. Stale cache hits are impossible by construction: change a parameter →
new hash → cache miss → stage runs.

This module exposes:
  - ArtifactPaths: resolves the canonical on-disk path for each stage's output.
  - cached(): the one-and-only stage wrapper. Loads from disk if present; else
    invokes the producer and writes the result atomically (temp + rename).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    """Resolves canonical cache paths for a single video.

    Hashes are caller-supplied (the pipeline computes them from stage inputs).
    """

    cache_dir: Path
    video_id: str

    @property
    def root(self) -> Path:
        return self.cache_dir / self.video_id

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def metadata_raw(self) -> Path:
        return self.root / "metadata.raw.json"

    def audio(self, *, compression_hash: str) -> Path:
        return self.root / f"audio-{compression_hash}.opus"

    def transcript(self, *, input_hash: str) -> Path:
        return self.root / f"transcript-{input_hash}.json"

    def transcript_raw(self, *, input_hash: str) -> Path:
        return self.root / f"transcript-{input_hash}.raw.json"

    def cleaned(self, *, input_hash: str) -> Path:
        return self.root / f"cleaned-{input_hash}.json"

    def structured(self, *, input_hash: str) -> Path:
        return self.root / f"structured-{input_hash}.json"
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_cache_paths.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/cache.py tests/unit/test_cache_paths.py
git commit -m "$(cat <<'EOF'
feat(cache): add ArtifactPaths with hash-keyed per-stage paths

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6.2: cached() helper — miss path (produce + write)

**Files:**
- Create: `tests/unit/test_cache_helper.py`
- Modify: `src/yt2md/cache.py` (append)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_cache_helper.py`:

```python
"""Tests for the cached() helper: miss path, hit path, atomicity."""

from pathlib import Path

import pytest

from yt2md.cache import cached


def _load_str(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _dump_str(value: str, p: Path) -> None:
    p.write_text(value, encoding="utf-8")


class TestCacheMiss:
    def test_miss_invokes_producer_and_returns_result(self, tmp_path: Path) -> None:
        target = tmp_path / "artifact.txt"
        calls: list[int] = []

        def produce() -> str:
            calls.append(1)
            return "hello"

        result = cached(path=target, produce=produce, load=_load_str, dump=_dump_str)
        assert result == "hello"
        assert len(calls) == 1

    def test_miss_writes_file_atomically(self, tmp_path: Path) -> None:
        target = tmp_path / "artifact.txt"
        cached(path=target, produce=lambda: "x", load=_load_str, dump=_dump_str)
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "x"

    def test_miss_creates_parent_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "deep" / "artifact.txt"
        cached(path=target, produce=lambda: "y", load=_load_str, dump=_dump_str)
        assert target.exists()

    def test_no_tmp_file_left_behind(self, tmp_path: Path) -> None:
        target = tmp_path / "artifact.txt"
        cached(path=target, produce=lambda: "z", load=_load_str, dump=_dump_str)
        leftover = list(tmp_path.glob("*.tmp"))
        assert leftover == []


class TestProducerFailureLeavesNoArtifact:
    def test_producer_raises_no_file_written(self, tmp_path: Path) -> None:
        target = tmp_path / "artifact.txt"

        def produce() -> str:
            msg = "boom"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="boom"):
            cached(path=target, produce=produce, load=_load_str, dump=_dump_str)
        assert not target.exists()
        leftover = list(tmp_path.glob("*"))
        assert leftover == []
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_cache_helper.py -v
```

Expected: FAIL — `cached` does not exist.

- [ ] **Step 3: Append to `src/yt2md/cache.py`**

Add imports at the top:

```python
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
```

Then append at the bottom:

```python
def cached(
    *,
    path: Path,
    produce: Callable[[], T],
    load: Callable[[Path], T],
    dump: Callable[[T, Path], None],
) -> T:
    """Read the artifact at `path` if present; else produce, write atomically, return.

    Atomicity: dump writes to a sibling `.tmp` file, which is then renamed via
    `Path.replace` (POSIX rename is atomic). On producer failure, no artifact is
    left behind.
    """
    if path.exists():
        return load(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        result = produce()
    except BaseException:
        # Producer failed before we wrote anything; ensure no stray tmp file.
        if tmp.exists():
            tmp.unlink()
        raise

    try:
        dump(result, tmp)
        tmp.replace(path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise

    return result
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_cache_helper.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/cache.py tests/unit/test_cache_helper.py
git commit -m "$(cat <<'EOF'
feat(cache): add cached() helper with atomic temp+rename and failure cleanup

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6.3: cached() — hit path (load, skip producer)

**Files:**
- Modify: `tests/unit/test_cache_helper.py` (append)

(No new implementation — the hit path is already covered by the existing `cached()`. This task adds the test that proves the producer is not called on hit.)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_cache_helper.py`:

```python
class TestCacheHit:
    def test_hit_skips_producer(self, tmp_path: Path) -> None:
        target = tmp_path / "artifact.txt"
        target.write_text("prewritten", encoding="utf-8")
        calls: list[int] = []

        def produce() -> str:
            calls.append(1)
            return "should-not-be-called"

        result = cached(path=target, produce=produce, load=_load_str, dump=_dump_str)
        assert result == "prewritten"
        assert calls == []

    def test_hit_returns_load_result(self, tmp_path: Path) -> None:
        target = tmp_path / "artifact.txt"
        target.write_text("from-disk", encoding="utf-8")
        result = cached(path=target, produce=lambda: "from-producer", load=_load_str, dump=_dump_str)
        assert result == "from-disk"
```

- [ ] **Step 2: Run — confirm passes**

```bash
uv run pytest tests/unit/test_cache_helper.py -v
```

Expected: all 7 tests PASS (5 existing + 2 new).

- [ ] **Step 3: Lint + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add tests/unit/test_cache_helper.py
git commit -m "$(cat <<'EOF'
test(cache): cover hit path skips producer and returns loaded value

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6.4: Fingerprint helper (short-hash for cache keys)

**Files:**
- Create: `tests/unit/test_cache_fingerprint.py`
- Modify: `src/yt2md/cache.py` (append)

The pipeline needs a stable short-hash for cache keys. This is a tiny, focused utility.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_cache_fingerprint.py`:

```python
"""Tests for the fingerprint() helper used to build cache keys."""

from yt2md.cache import fingerprint


class TestFingerprint:
    def test_deterministic(self) -> None:
        a = fingerprint("opus", 32, "mono")
        b = fingerprint("opus", 32, "mono")
        assert a == b

    def test_order_sensitive(self) -> None:
        a = fingerprint("opus", 32)
        b = fingerprint(32, "opus")
        assert a != b

    def test_different_inputs_different_hash(self) -> None:
        a = fingerprint("opus", 32)
        b = fingerprint("opus", 64)
        assert a != b

    def test_length_is_short(self) -> None:
        # Short enough for filenames; long enough to avoid trivial collisions.
        h = fingerprint("anything")
        assert 8 <= len(h) <= 16

    def test_alphanumeric_only(self) -> None:
        h = fingerprint("anything", 1, None, [1, 2, 3])
        assert h.isalnum()

    def test_supports_none(self) -> None:
        h = fingerprint(None, "x")
        assert h
        assert h.isalnum()

    def test_supports_lists(self) -> None:
        h = fingerprint([1, 2, 3])
        assert h
        assert h.isalnum()
```

- [ ] **Step 2: Run — confirm fails**

```bash
uv run pytest tests/unit/test_cache_fingerprint.py -v
```

Expected: FAIL — `fingerprint` does not exist.

- [ ] **Step 3: Append to `src/yt2md/cache.py`**

Add at the top imports:

```python
import hashlib
import json
from typing import Any
```

Append at the bottom:

```python
FINGERPRINT_LENGTH = 12


def fingerprint(*parts: Any) -> str:
    """Deterministic short hash of the given parts, suitable for cache keys in filenames.

    Order-sensitive. Stable across processes and Python versions (uses SHA-256 over a
    canonical JSON encoding of the parts tuple).
    """
    canonical = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return digest[:FINGERPRINT_LENGTH]
```

- [ ] **Step 4: Run — confirm passes**

```bash
uv run pytest tests/unit/test_cache_fingerprint.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
uv run ruff check src/ tests/
uv run mypy --strict src/
git add src/yt2md/cache.py tests/unit/test_cache_fingerprint.py
git commit -m "$(cat <<'EOF'
feat(cache): add fingerprint() short-hash helper for cache keys

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1 final check

### Task 7.1: Run the full Phase 1 suite + coverage gate

- [ ] **Step 1: Run all tests with coverage**

```bash
uv run pytest tests/unit --cov=src/yt2md --cov-report=term-missing --cov-fail-under=85
```

Expected: all tests pass; coverage ≥85% across `errors.py`, `models.py`, `config.py`, `costs.py`, `cache.py`.

If coverage is under 85%, identify the uncovered lines (`term-missing` shows them) and add a test that exercises them. Likely candidates: validator error paths in models, edge branches in `cached()`.

- [ ] **Step 2: Run the full pre-commit suite**

```bash
uv run pre-commit run --all-files
```

Expected: all hooks pass.

- [ ] **Step 3: Confirm CI green**

Push to a feature branch and confirm the GitHub Actions workflow passes. (If not pushing, this step is N/A.)

- [ ] **Step 4: Mark Phase 1 complete in the index**

Edit `docs/superpowers/plans/2026-05-23-yt2llm-index.md` and check off Phase 1:

```markdown
- [x] Phase 1 — Bootstrap + Foundations
```

Commit:

```bash
git add docs/superpowers/plans/2026-05-23-yt2llm-index.md
git commit -m "$(cat <<'EOF'
docs(plan): mark Phase 1 complete

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## What Phase 1 produced

After all tasks above:

- `pyproject.toml` with full tooling stack (uv-managed, Ruff `ALL` + tight PLR thresholds, mypy `--strict`, pytest, hypothesis).
- `.pre-commit-config.yaml` enforcing lint, format, typecheck, 400-LOC ceiling, fast unit tests on every commit.
- `.github/workflows/ci.yml` running lint + typecheck + tests with 85% coverage gate.
- `src/yt2md/errors.py` — typed exception hierarchy.
- `src/yt2md/models.py` — full Pydantic v2 data contracts.
- `src/yt2md/config.py` — pydantic-settings with 12-factor precedence.
- `src/yt2md/costs.py` — per-model cost calculators.
- `src/yt2md/cache.py` — `ArtifactPaths`, `cached()`, `fingerprint()`.
- A green test suite for all of the above with ≥85% coverage.

**No external API calls, no filesystem beyond tmp_path, no orchestration logic.** Phase 2 builds the deterministic stages on top of this foundation.

---

## Next: Phase 2

Open `docs/superpowers/plans/2026-05-23-yt2llm-phase-2-deterministic-stages.md` (write it after Phase 1 lands, or in parallel if you want it ready in advance — ask your AI helper to produce it).
