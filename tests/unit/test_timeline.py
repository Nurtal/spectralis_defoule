import numpy as np
import pytest

from conversation_deconvolution.core.types import Segment, SpeakerTurn
from conversation_deconvolution.diarization.clusterer import AgglomerativeClusterer
from conversation_deconvolution.diarization.timeline import merge_turns, overlap_regions


def test_merge_consecutive_same_speaker():
    turns = [SpeakerTurn("A", 0, 1), SpeakerTurn("A", 1.05, 2), SpeakerTurn("B", 2.5, 3)]
    merged = merge_turns(turns, gap=0.2)
    assert len(merged) == 2
    assert merged[0] == SpeakerTurn("A", 0, 2)
    assert merged[1] == SpeakerTurn("B", 2.5, 3)


def test_merge_keeps_distant_same_speaker():
    turns = [SpeakerTurn("A", 0, 1), SpeakerTurn("A", 5, 6)]
    assert len(merge_turns(turns, gap=0.2)) == 2


def test_overlap_basic():
    turns = [SpeakerTurn("A", 0, 2), SpeakerTurn("B", 1, 3)]
    assert overlap_regions(turns) == [Segment(1.0, 2.0)]


def test_overlap_disjoint_empty():
    turns = [SpeakerTurn("A", 0, 1), SpeakerTurn("B", 2, 3)]
    assert overlap_regions(turns) == []


def test_clusterer_separates_two_blobs_without_k():
    rng = np.random.default_rng(0)
    d = np.zeros(16); d[0] = 3.0
    e = np.zeros(16); e[1] = 3.0
    blob_a = d + rng.normal(0, 0.05, size=(12, 16))
    blob_b = e + rng.normal(0, 0.05, size=(12, 16))
    X = np.vstack([blob_a, blob_b])
    labels = AgglomerativeClusterer(distance_threshold=0.5).fit_predict(X)
    assert len(set(labels)) == 2
    assert len(set(labels[:12])) == 1 and len(set(labels[12:])) == 1


def test_clusterer_respects_k():
    rng = np.random.default_rng(1)
    X = np.vstack([
        rng.normal(0, 0.03, size=(6, 16)),
        rng.normal(2.0, 0.03, size=(6, 16)),
        rng.normal(4.0, 0.03, size=(6, 16)),
    ])
    labels = AgglomerativeClusterer().fit_predict(X, n_speakers=3)
    assert len(set(labels)) == 3


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_clusterer_single_sample():
    labels = AgglomerativeClusterer().fit_predict(np.ones((1, 8)))
    assert labels.tolist() == [0]
