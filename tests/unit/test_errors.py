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
