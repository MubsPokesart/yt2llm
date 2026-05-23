"""Tests for mapping yt-dlp DownloadError strings to typed exceptions."""

import pytest

from yt2md.errors import LivestreamNotEndedError, VideoUnavailableError, YT2MDError
from yt2md.stages.download import map_ytdlp_error


class TestErrorMapping:
    @pytest.mark.parametrize(
        "ytdlp_message",
        [
            "ERROR: [youtube] Private video. Sign in if you've been granted access",
            "ERROR: [youtube] Video unavailable",
            "ERROR: [youtube] This video has been removed",
        ],
    )
    def test_private_or_removed(self, ytdlp_message: str) -> None:
        err = map_ytdlp_error(Exception(ytdlp_message))
        assert isinstance(err, VideoUnavailableError)

    def test_age_restricted_includes_cookie_hint(self) -> None:
        err = map_ytdlp_error(
            Exception(
                "ERROR: [youtube] Sign in to confirm your age. This video may be inappropriate"
            )
        )
        assert isinstance(err, VideoUnavailableError)
        assert "cookies-from-browser" in str(err).lower()

    def test_members_only_includes_cookie_hint(self) -> None:
        err = map_ytdlp_error(
            Exception("ERROR: [youtube] Join this channel to get access to members-only content")
        )
        assert isinstance(err, VideoUnavailableError)
        assert "cookies" in str(err).lower()

    def test_geoblocked(self) -> None:
        err = map_ytdlp_error(
            Exception(
                "ERROR: [youtube] The uploader has not made this video available in your country"
            )
        )
        assert isinstance(err, VideoUnavailableError)

    def test_livestream_not_ended(self) -> None:
        err = map_ytdlp_error(Exception("ERROR: This live event will begin in 2 hours"))
        assert isinstance(err, LivestreamNotEndedError)

    def test_unmatched_falls_back_to_root(self) -> None:
        err = map_ytdlp_error(Exception("Some entirely unknown error string"))
        assert isinstance(err, YT2MDError)
