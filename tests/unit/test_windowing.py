
import pytest

from conversation_deconvolution.core.types import Segment
from conversation_deconvolution.diarization.timeline import (
    make_windows,
    margin_regions,
    windows_to_turns,
)


def test_make_windows_covers_segment():
    wins = make_windows([Segment(0, 4.0)], window_sec=1.5, hop_sec=0.5)
    assert wins[0][0] == 0.0
    assert wins[-1][1] == pytest.approx(4.0)
    for ws, we in wins:
        assert we > ws and we - ws <= 1.5 + 1e-9


def test_make_windows_short_segment_untouched():
    assert make_windows([Segment(1.0, 2.0)], window_sec=1.5, hop_sec=0.5) == [(1.0, 2.0)]


def test_windows_to_turns_empty():
    assert windows_to_turns([], []) == []


def test_single_window_single_turn():
    turns = windows_to_turns([(0.0, 1.5)], [3])
    assert turns == [(3, 0.0, 1.5)]


def test_alternating_windows_split_at_boundary():
    turns = windows_to_turns([(0.0, 2.0), (2.0, 4.0)], [0, 1], min_turn_sec=0.1)
    assert [(lab, s, e) for lab, s, e in turns] == [
        (0, 0.0, pytest.approx(2.0)),
        (1, pytest.approx(2.0), pytest.approx(4.0)),
    ]


def test_majority_vote_with_overlap_and_tie_prefers_previous():
    wins = [(0.0, 1.5), (0.5, 2.0), (1.5, 3.0)]
    labels = [0, 0, 1]
    turns = windows_to_turns(wins, labels, cell_sec=0.25, min_turn_sec=0.1)
    assert [(lab, round(s, 2), round(e, 2)) for lab, s, e in turns] == [
        (0, 0.0, 2.0),
        (1, 2.0, 3.0),
    ]


def test_short_spurious_run_absorbed():
    wins = [(i * 0.4, (i + 1) * 0.4) for i in range(4)]
    labels = [0, 0, 1, 0]
    turns = windows_to_turns(wins, labels, cell_sec=0.1, min_turn_sec=0.5)
    assert len(turns) == 1
    assert turns[0][0] == 0


def test_margin_regions_detect_contested_zone():
    # windows 0-1s and 2-3s are cleanly owned; 1-2s alternates owners with
    # near-evidence -> contested zone flagged
    wins = [
        (0.0, 1.0),
        (0.5, 1.5),
        (0.5, 1.5),
        (1.0, 2.0),
        (1.0, 2.0),
        (1.5, 2.5),
        (1.5, 2.5),
        (2.0, 3.0),
    ]
    labels = [0, 0, 1, 1, 0, 0, 1, 0]
    regions = margin_regions(
        wins, labels, cell_sec=0.25, min_margin=0.34, min_duration=0.3
    )
    # contradictory window pairs span [0.5, 2.5]
    assert regions == [Segment(0.5, 2.5)]


def test_margin_regions_clean_track_has_none():
    wins = [(i * 1.0, (i + 1) * 1.0) for i in range(4)]
    labels = [0, 0, 1, 1]
    assert (
        margin_regions(wins, labels, cell_sec=0.25, min_margin=0.34, min_duration=0.3)
        == []
    )
