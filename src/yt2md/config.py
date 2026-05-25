"""Configuration loading for yt2llm.

Precedence (12-factor): CLI flag > env var > TOML file > default.

env_prefix is YT2MD_. So YT2MD_OUTPUT_DIR sets output_dir.
TOML files searched (later wins): ~/.config/yt2md/config.toml, ./yt2md.toml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr  # noqa: TC002  -- Pydantic needs runtime import
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
