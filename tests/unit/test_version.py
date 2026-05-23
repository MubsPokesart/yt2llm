"""Smoke test: package version is importable."""

import yt2md


def test_version_is_a_string() -> None:
    assert isinstance(yt2md.__version__, str)


def test_version_is_nonempty() -> None:
    assert yt2md.__version__
