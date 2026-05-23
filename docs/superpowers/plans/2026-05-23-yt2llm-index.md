# yt2llm Implementation Plan — Index

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement each phase task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-05-23-yt2llm-design.md`

**Goal:** Build the yt2llm MVP — a Python CLI that turns any YouTube URL into a Tier 3 structured markdown file, with strict TDD + linting + decomposition discipline enforced from day 1.

**Architecture:** Seven-stage pipeline with on-disk artifact caching, plain typed functions as stages, Pydantic v2 data contracts, Gemini structured output with retry, faster-whisper local fallback, idempotent CLI.

**Tech Stack:** Python 3.11+, uv, Ruff (ALL with tight PLR thresholds), mypy --strict, pre-commit, pytest, hypothesis, Pydantic v2, pydantic-settings, typer, tenacity, rich, structlog, Jinja2, tiktoken, yt-dlp, openai, google-genai, faster-whisper (optional).

---

## Phase ordering and dependencies

Phases are sequential — each depends on the prior. **Do not begin a phase until the prior phase's tests are green and committed.**

```
Phase 1 (Foundations)
    ↓
Phase 2 (Deterministic stages)
    ↓
Phase 3 (API-bound stages)
    ↓
Phase 4 (Orchestrator + CLI)
```

| Phase | File | What it produces |
|---|---|---|
| **1** | `2026-05-23-yt2llm-phase-1-foundations.md` | Project skeleton, tooling, infra modules (`errors`, `models`, `config`, `costs`, `cache`). Tests pass; CI green. |
| **2** | `2026-05-23-yt2llm-phase-2-deterministic-stages.md` | `vocab_hint`, `clean`, `render`, `write`, Jinja2 templates. Round-trip a fixture `StructuredDoc` to golden markdown. |
| **3** | `2026-05-23-yt2llm-phase-3-api-stages.md` | `download`, `compress`, `transcribe` (both backends + chunking), `structure`. Each stage tested with mocks + (optional) live. |
| **4** | `2026-05-23-yt2llm-phase-4-orchestrator-cli.md` | `pipeline.run`, `cli` (typer), idempotency, `regen` subcommand, observability glue (rich + structlog + runs.log), end-to-end integration tests. MVP shippable. |

---

## Non-negotiable discipline (every phase, every task)

These are repeated at the top of each phase plan. They are not aspirational.

### TDD red-green-refactor

Every task follows the same shape:
1. Write the failing test (with full code shown).
2. Run it — confirm it fails for the expected reason.
3. Write the minimum implementation to make it pass.
4. Run tests — confirm green.
5. Run `ruff check src/ tests/` and `mypy --strict src/` — confirm green.
6. Commit.

**No production code without a corresponding test.** No exceptions for "load-bearing code." If a step's test passes immediately on the failure-check, that's a bug — the test wasn't testing what you thought.

### Lint + type enforcement

Every commit must pass:
```
ruff check src/ tests/
ruff format --check src/ tests/
mypy --strict src/
pytest -q
```

The pre-commit hook configured in Phase 1 enforces this automatically. Disabling the hook (`--no-verify`) is **never** permitted. If a hook is failing, fix the code or the test, not the hook.

### Decomposition

- **400 LOC per `src/` module** ceiling (enforced by pre-commit hook).
- **5 args, 30 statements, 8 branches, complexity 10, 4 nested blocks** per function (enforced by Ruff PLR rules).
- A new abstraction (wrapper, base class, decorator, protocol, helper function) requires **≥3 concrete call sites**. Two is a coincidence; three is a pattern.
- **No code for hypothetical future use.** If a task description mentions it, build it. If it doesn't, don't.

### Commit hygiene

- One coherent change per commit.
- Commit messages are imperative present-tense ("add transcript cache", not "added").
- The failing-test commit and the implementation commit may be the same commit (atomic red-green), or two commits in sequence. Either is fine; both are not optional.
- Never `git commit --no-verify`. Never `git commit --amend` to bypass hooks.

---

## Conventions used across phase plans

To keep tasks dense, the following are defined once here and not repeated.

### Shorthand commands

| Shorthand | Full command |
|---|---|
| `lint` | `ruff check src/ tests/ && ruff format --check src/ tests/` |
| `typecheck` | `mypy --strict src/` |
| `test <path>` | `pytest <path> -v` |
| `testall` | `pytest tests/unit tests/integration -q` |
| `cover` | `pytest tests/unit tests/integration --cov=src/yt2md --cov-report=term-missing --cov-fail-under=85` |

These are aliases used in plan steps. They're not shell aliases; they map to the full commands above.

### Commit message format

```
<type>(<scope>): <imperative summary>
```

Types: `feat`, `test`, `refactor`, `chore`, `docs`, `fix`, `ci`.

Scope is the module or area (e.g., `cache`, `clean`, `transcribe-openai`, `cli`).

Examples used in this plan:
- `feat(cache): add cached() helper with atomic temp+rename`
- `test(clean): cover HARD_FILLERS removal preserving timestamps`
- `chore(tooling): pin ruff to 0.X.Y in pre-commit config`

### Co-author trailer

Every commit includes:
```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

(Skip this if you, the engineer, are working solo without AI assistance.)

### Path notation

All file paths in plans are **absolute** within the project (`src/yt2md/...`, `tests/...`). No `./` prefixes. No tilde expansions.

### Code blocks

- Code shown in plan steps is what gets written verbatim, with the noted exception below.
- For very long fixture files (>100 lines), the plan shows the shape with a representative excerpt and notes "extend pattern for full content."
- Test code is always shown in full — never elided.

---

## Execution choice (per phase)

After completing each phase, choose:

**1. Subagent-Driven (recommended for next phase)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Use `superpowers:subagent-driven-development`.

**2. Inline Execution** — execute tasks in the current session. Use `superpowers:executing-plans`. Batch with checkpoints for review.

---

## Status tracking

Mark phases here as you complete them:

- [x] Phase 1 — Bootstrap + Foundations
- [x] Phase 2 — Deterministic stages
- [x] Phase 3 — API-bound stages
- [x] Phase 4 — Orchestrator + CLI

Each phase plan has its own checkbox list of tasks. Mark off as you go.

---

**Begin with Phase 1: `docs/superpowers/plans/2026-05-23-yt2llm-phase-1-foundations.md`.**
