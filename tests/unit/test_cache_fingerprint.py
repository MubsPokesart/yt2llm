"""Tests for the fingerprint() helper used to build cache keys."""

from yt2md.cache import fingerprint


class TestFingerprint:
    def test_deterministic(self) -> None:
        a = fingerprint("opus", 32, "mono")
        b = fingerprint("opus", 32, "mono")
        assert a == b

    def test_order_sensitive(self) -> None:
        a = fingerprint("opus", 32)
        b = fingerprint(32, "opus")
        assert a != b

    def test_different_inputs_different_hash(self) -> None:
        a = fingerprint("opus", 32)
        b = fingerprint("opus", 64)
        assert a != b

    def test_length_is_short(self) -> None:
        # Short enough for filenames; long enough to avoid trivial collisions.
        h = fingerprint("anything")
        assert 8 <= len(h) <= 16

    def test_alphanumeric_only(self) -> None:
        h = fingerprint("anything", 1, None, [1, 2, 3])
        assert h.isalnum()

    def test_supports_none(self) -> None:
        h = fingerprint(None, "x")
        assert h
        assert h.isalnum()

    def test_supports_lists(self) -> None:
        h = fingerprint([1, 2, 3])
        assert h
        assert h.isalnum()
