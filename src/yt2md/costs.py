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
