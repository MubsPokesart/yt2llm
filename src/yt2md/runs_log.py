"""Append-only JSONL writer for runs.log — one line per pipeline invocation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

COST_ROUND_PRECISION = 4


@dataclass(frozen=True)
class RunRecord:
    """One pipeline invocation's outcome. Serialized as a JSONL line."""

    video_id: str
    url: str
    status: str  # "success" | "skipped" | "failed"
    duration_s: float
    transcription_usd: float
    structuring_usd: float
    transcription_backend: str
    cache_hits: list[str]
    stages_run: list[str]
    audio_mb: float
    video_duration_s: float
    schema_version: int
    error_class: str | None
    error_message: str | None


def append_run(log_path: Path, record: RunRecord) -> None:
    """Append one record as a JSONL line. Creates parent directory if missing."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_payload(record)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _to_payload(record: RunRecord) -> dict[str, object]:
    """Convert RunRecord into the public JSONL payload shape (costs grouped)."""
    raw = asdict(record)
    transcription = raw.pop("transcription_usd")
    structuring = raw.pop("structuring_usd")
    backend = raw.pop("transcription_backend")
    raw["costs"] = {
        "transcription_usd": transcription,
        "structuring_usd": structuring,
        "transcription_backend": backend,
        "total_usd": round(transcription + structuring, COST_ROUND_PRECISION),
    }
    raw["ts"] = datetime.now(UTC).isoformat()
    return raw
