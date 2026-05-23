"""Property test: clean() never invents timestamps.

Every surviving word's (start, end) tuple must have been in the input transcript.
"""

from hypothesis import given
from hypothesis import strategies as st

from yt2md.models import Segment, Transcript, Word
from yt2md.stages.clean import clean


def _word_from_components(
    text: str,
    interval: tuple[float, float],
    speaker: str | None,
) -> Word:
    start, end = sorted(interval)
    return Word(text=text, start=start, end=end, speaker=speaker)


_interval_strategy = st.tuples(
    st.floats(min_value=0.0, max_value=100.0),
    st.floats(min_value=0.0, max_value=100.0),
)

_word_strategy = st.builds(
    _word_from_components,
    text=st.text(min_size=1, max_size=10).filter(lambda s: not s.isspace()),
    interval=_interval_strategy,
    speaker=st.sampled_from(["S0", "S1", None]),
)


def _segments_from_words(words: list[Word]) -> list[Segment]:
    if not words:
        return []
    return [
        Segment(
            start=words[0].start,
            end=max(w.end for w in words),
            text=" ".join(w.text for w in words),
            speaker=words[0].speaker,
            words=words,
        ),
    ]


@given(words=st.lists(_word_strategy, min_size=1, max_size=20))
def test_surviving_timestamps_are_subset_of_input(words: list[Word]) -> None:
    segments = _segments_from_words(words)
    duration = max(w.end for w in words)
    t = Transcript(
        language="en",
        duration_s=duration,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=segments,
        speakers=["S0", "S1"] if any(w.speaker for w in words) else [],
    )
    result = clean(t)
    input_pairs = {(w.start, w.end) for w in words}
    output_pairs = {(w.start, w.end) for s in result.segments for w in s.words}
    assert output_pairs.issubset(input_pairs)
