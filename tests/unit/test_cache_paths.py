"""Tests for ArtifactPaths path resolution."""

from pathlib import Path

from yt2md.cache import ArtifactPaths


class TestArtifactPaths:
    def test_root(self, tmp_path: Path) -> None:
        paths = ArtifactPaths(cache_dir=tmp_path, video_id="abc123")
        assert paths.root == tmp_path / "abc123"

    def test_metadata(self, tmp_path: Path) -> None:
        paths = ArtifactPaths(cache_dir=tmp_path, video_id="abc")
        assert paths.metadata == tmp_path / "abc" / "metadata.json"

    def test_metadata_raw(self, tmp_path: Path) -> None:
        paths = ArtifactPaths(cache_dir=tmp_path, video_id="abc")
        assert paths.metadata_raw == tmp_path / "abc" / "metadata.raw.json"

    def test_audio_includes_compression_hash(self, tmp_path: Path) -> None:
        paths = ArtifactPaths(cache_dir=tmp_path, video_id="abc")
        p = paths.audio(compression_hash="deadbeef")
        assert p == tmp_path / "abc" / "audio-deadbeef.opus"

    def test_transcript_includes_input_hash(self, tmp_path: Path) -> None:
        paths = ArtifactPaths(cache_dir=tmp_path, video_id="abc")
        p = paths.transcript(input_hash="cafebabe")
        assert p == tmp_path / "abc" / "transcript-cafebabe.json"

    def test_transcript_raw(self, tmp_path: Path) -> None:
        paths = ArtifactPaths(cache_dir=tmp_path, video_id="abc")
        p = paths.transcript_raw(input_hash="cafebabe")
        assert p == tmp_path / "abc" / "transcript-cafebabe.raw.json"

    def test_cleaned(self, tmp_path: Path) -> None:
        paths = ArtifactPaths(cache_dir=tmp_path, video_id="abc")
        p = paths.cleaned(input_hash="aabb")
        assert p == tmp_path / "abc" / "cleaned-aabb.json"

    def test_structured(self, tmp_path: Path) -> None:
        paths = ArtifactPaths(cache_dir=tmp_path, video_id="abc")
        p = paths.structured(input_hash="ccdd")
        assert p == tmp_path / "abc" / "structured-ccdd.json"
