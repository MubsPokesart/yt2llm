"""Live test for download() — hits real YouTube. Skipped without `-m live`."""

from pathlib import Path

import pytest

from yt2md.config import Config
from yt2md.stages.download import download

# A short, permanently-public, copyright-free test video (19 seconds).
# "Me at the zoo" — the very first YouTube video, uploaded by jawed in 2005.
STABLE_TEST_URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


@pytest.mark.live
def test_live_download_real_youtube_video(tmp_path: Path) -> None:
    cfg = Config(
        google_api_key="g",  # type: ignore[arg-type]
        cache_dir=tmp_path,
        output_dir=tmp_path / "out",
    )
    audio_path, meta, raw = download(STABLE_TEST_URL, cfg=cfg)
    assert audio_path.exists()
    assert audio_path.stat().st_size > 1000  # at least a few KB
    assert meta.duration_s > 5  # 19s video
    assert meta.video_id == "jNQXAC9IVRw"
    assert isinstance(raw, dict)
