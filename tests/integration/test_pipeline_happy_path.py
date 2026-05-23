"""End-to-end pipeline integration test with all stages mocked at the module boundary."""

from typing import Any

from yt2md.config import Config
from yt2md.pipeline import run


class TestPipelineRun:
    def test_returns_path_to_written_markdown(self, cfg: Config, patched_stages: Any) -> None:
        url = "https://www.youtube.com/watch?v=abc123"
        path = run(url, cfg=cfg)
        assert path.exists()
        assert path.suffix == ".md"
        assert "test-episode" in path.name

    def test_markdown_content_includes_frontmatter(self, cfg: Config, patched_stages: Any) -> None:
        path = run("https://www.youtube.com/watch?v=abc123", cfg=cfg)
        content = path.read_text(encoding="utf-8")
        assert content.startswith("---")
        assert "title:" in content
        assert "Test Episode" in content

    def test_cache_populated(self, cfg: Config, patched_stages: Any) -> None:
        run("https://www.youtube.com/watch?v=abc123", cfg=cfg)
        cache_subdir = cfg.cache_dir / "abc123"
        assert cache_subdir.exists()
        assert (cache_subdir / "metadata.json").exists()
        assert any(cache_subdir.glob("transcript-*.json"))
        assert any(cache_subdir.glob("cleaned-*.json"))
        assert any(cache_subdir.glob("structured-*.json"))


class TestPipelineResume:
    def test_second_run_with_cache_skips_upstream_stages(
        self, cfg: Config, patched_stages: Any
    ) -> None:
        url = "https://www.youtube.com/watch?v=abc123"
        run(url, cfg=cfg)
        patched_stages["dl"].reset_mock()
        patched_stages["cmp"].reset_mock()
        patched_stages["tx"].reset_mock()
        patched_stages["st"].reset_mock()

        out_files = list(cfg.output_dir.glob("*.md"))
        for f in out_files:
            f.unlink()

        run(url, cfg=cfg)
        assert patched_stages["dl"].call_count == 0
        assert patched_stages["cmp"].call_count == 0
        assert patched_stages["tx"].call_count == 0
        assert patched_stages["st"].call_count == 0
