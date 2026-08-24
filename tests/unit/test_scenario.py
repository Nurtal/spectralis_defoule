import numpy as np

from conversation_deconvolution.core.types import Utterance
from conversation_deconvolution.synthetic.scenario import generate_scenario


def test_scenario_deterministic_with_seed():
    s1 = generate_scenario(n_conversations=2, speakers_per_thread=2, n_lines=(4, 6), rng=np.random.default_rng(42))
    s2 = generate_scenario(n_conversations=2, speakers_per_thread=2, n_lines=(4, 6), rng=np.random.default_rng(42))
    assert [l.text for t in s1 for l in t.lines] == [l.text for t in s2 for l in t.lines]
    assert [g for t in s1 for g in t.gaps] == [g for t in s2 for g in t.gaps]


def test_scenario_counts_and_voices():
    threads = generate_scenario(
        n_conversations=3, speakers_per_thread=2, n_lines=(4, 8), rng=np.random.default_rng(1)
    )
    assert len(threads) == 3
    for thread in threads:
        assert len(thread.speakers) == 2
        assert all(len(l.text.split()) >= 2 for l in thread.lines)
        voices = {l.voice for l in thread.lines if l.speaker == thread.speakers[0]}
        assert len(voices) == 1


def test_gaps_positive():
    threads = generate_scenario(
        n_conversations=1, speakers_per_thread=2, n_lines=(5, 5), rng=np.random.default_rng(7)
    )
    assert all(g >= 0.15 for g in threads[0].gaps)
