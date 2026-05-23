"""Tests for the cached() helper: miss path, hit path, atomicity."""

from pathlib import Path

import pytest

from yt2md.cache import cached


def _load_str(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _dump_str(value: str, p: Path) -> None:
    p.write_text(value, encoding="utf-8")


class TestCacheMiss:
    def test_miss_invokes_producer_and_returns_result(self, tmp_path: Path) -> None:
        target = tmp_path / "artifact.txt"
        calls: list[int] = []

        def produce() -> str:
            calls.append(1)
            return "hello"

        result = cached(path=target, produce=produce, load=_load_str, dump=_dump_str)
        assert result == "hello"
        assert len(calls) == 1

    def test_miss_writes_file_atomically(self, tmp_path: Path) -> None:
        target = tmp_path / "artifact.txt"
        cached(path=target, produce=lambda: "x", load=_load_str, dump=_dump_str)
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "x"

    def test_miss_creates_parent_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "deep" / "artifact.txt"
        cached(path=target, produce=lambda: "y", load=_load_str, dump=_dump_str)
        assert target.exists()

    def test_no_tmp_file_left_behind(self, tmp_path: Path) -> None:
        target = tmp_path / "artifact.txt"
        cached(path=target, produce=lambda: "z", load=_load_str, dump=_dump_str)
        leftover = list(tmp_path.glob("*.tmp"))
        assert leftover == []


class TestProducerFailureLeavesNoArtifact:
    def test_producer_raises_no_file_written(self, tmp_path: Path) -> None:
        target = tmp_path / "artifact.txt"

        def produce() -> str:
            msg = "boom"
            raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="boom"):
            cached(path=target, produce=produce, load=_load_str, dump=_dump_str)
        assert not target.exists()
        leftover = list(tmp_path.glob("*"))
        assert leftover == []
