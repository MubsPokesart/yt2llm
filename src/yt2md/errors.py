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
    """Video cannot be accessed: private, removed, age-restricted, members-only, geoblocked.

    Exit code 2.
    """


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
