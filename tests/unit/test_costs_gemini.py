"""Tests for Gemini cost calculator."""

import pytest

from yt2md.costs import gemini_flash_cost


class TestGeminiFlash:
    def test_one_million_input(self) -> None:
        # $0.50 per 1M input tokens
        cost = gemini_flash_cost(input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(0.50, rel=1e-3)

    def test_one_million_output(self) -> None:
        cost = gemini_flash_cost(input_tokens=0, output_tokens=1_000_000)
        assert cost == pytest.approx(3.00, rel=1e-3)

    def test_mixed(self) -> None:
        cost = gemini_flash_cost(input_tokens=25_000, output_tokens=10_000)
        # 25k * 0.50/1M = 0.0125; 10k * 3.00/1M = 0.030; total 0.0425
        assert cost == pytest.approx(0.0425, rel=1e-3)

    def test_zero(self) -> None:
        assert gemini_flash_cost(input_tokens=0, output_tokens=0) == pytest.approx(0.0)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            gemini_flash_cost(input_tokens=-1, output_tokens=0)
        with pytest.raises(ValueError, match="non-negative"):
            gemini_flash_cost(input_tokens=0, output_tokens=-1)
