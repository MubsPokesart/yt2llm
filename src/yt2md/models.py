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
