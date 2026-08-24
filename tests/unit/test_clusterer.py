import numpy as np

from conversation_deconvolution.diarization.clusterer import AgglomerativeClusterer


def _blobs(angles_deg, n_per=20, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    vecs = []
    for ang in angles_deg:
        v = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        vecs.append(v + rng.normal(0, noise, (n_per, 2)))
    X = np.vstack(vecs)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_fixed_k_respected():
    X = np.vstack([np.ones((5, 1)), -np.ones((5, 1))])
    labels = AgglomerativeClusterer().fit_predict(X, n_speakers=2)
    assert len(set(labels)) == 2


def test_single_sample():
    assert list(AgglomerativeClusterer().fit_predict(np.ones((1, 4)))) == [0]


def test_auto_k_recovers_close_blobs_threshold_misses():
    # pairwise cosine distances ~0.4/0.4/0.8: fixed threshold 0.75 merges all,
    # silhouette selection must still recover 3 clusters.
    X = _blobs([0, 55, 110])
    labels = AgglomerativeClusterer(distance_threshold=0.75).fit_predict(X)
    assert len(set(labels)) == 3


def test_auto_k_two_samples_falls_back():
    labels = AgglomerativeClusterer().fit_predict(np.array([[1.0, 0], [-1.0, 0]]))
    assert len(labels) == 2


def test_auto_k_caps_max_speakers():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (40, 8))
    labels = AgglomerativeClusterer(max_speakers=3).fit_predict(X)
    assert len(set(labels)) <= 3
