"""Configuration loading for yt2llm.

Precedence: CLI flag > env var > default.

Env vars use the YT2MD_ prefix (e.g. YT2MD_OUTPUT_DIR sets output_dir). API key
fields also accept the bare OPENAI_API_KEY / GOOGLE_API_KEY names so users can
follow the OpenAI / Google docs verbatim; YT2MD_-prefixed values win on collision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YT2MD_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    # API keys — accept both the project-prefixed name (wins on collision) and the
    # convention bare name that OpenAI / Google docs tell users to export.
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("YT2MD_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    google_api_key: SecretStr = Field(
        validation_alias=AliasChoices("YT2MD_GOOGLE_API_KEY", "GOOGLE_API_KEY"),
    )

    # Paths
    output_dir: Path = Path("./output")
    cache_dir: Path = Path("./cache")

    # Audio
    audio_bitrate_kbps: int = 32
    audio_codec: Literal["opus"] = "opus"

    # Transcription
    transcription_backend: Literal["openai_transcribe", "local_whisper", "auto"] = "auto"
    transcription_model: str = "whisper-1"
    local_whisper_model: str = "medium"
    use_transcription_hint: bool = True

    # Structuring
    # gemini-2.5-flash: workhorse model, well-priced, handles structured output.
    # gemini-3-flash is preview-only on v1beta consumer keys (404). gemini-3.5-flash
    # is materially pricier without commensurate quality gains for this workload.
    structuring_model: str = "gemini-2.5-flash"

    # yt-dlp auth
    cookies_from_browser: str | None = None
    cookies_file: Path | None = None

    # CLI behavior flags
    force: bool = False
    no_cache: bool = False
