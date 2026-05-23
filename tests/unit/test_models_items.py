"""Tests for structured-doc item models: Reference, Takeaway, Concept, Quote, DetailedSection."""

import pytest
from pydantic import ValidationError

from yt2md.models import Concept, DetailedSection, Quote, Reference, Takeaway


class TestTakeaway:
    def test_takeaway(self) -> None:
        t = Takeaway(text="Dopamine signals anticipation.", timestamp_s=252.0)
        assert t.timestamp_s == pytest.approx(252.0)

    def test_negative_timestamp_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Takeaway(text="x", timestamp_s=-1.0)


class TestConcept:
    def test_concept(self) -> None:
        c = Concept(
            name="Reward Prediction Error",
            definition="Gap between expected and actual reward.",
            timestamp_s=510.0,
        )
        assert c.name == "Reward Prediction Error"


class TestReference:
    @pytest.mark.parametrize(
        "kind",
        ["book", "paper", "person", "tool", "video", "other"],
    )
    def test_reference_kinds(self, kind: str) -> None:
        r = Reference(kind=kind, name="X", context="ctx", timestamp_s=0.0)  # type: ignore[arg-type]
        assert r.kind == kind

    def test_invalid_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Reference(kind="movie", name="X", context="c", timestamp_s=0.0)  # type: ignore[arg-type]


class TestQuote:
    def test_quote(self) -> None:
        q = Quote(text="Pursuit, not pleasure.", speaker="Andrew Huberman", timestamp_s=754.0)
        assert q.speaker == "Andrew Huberman"


class TestDetailedSection:
    def test_section(self) -> None:
        s = DetailedSection(
            heading="What dopamine actually does", body="Multi paragraph.", timestamp_s=0.0
        )
        assert s.heading.startswith("What")
