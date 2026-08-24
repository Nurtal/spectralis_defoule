import hashlib

import numpy as np
import pytest

from conversation_deconvolution.conversation.reconstructor import HeuristicReconstructor
from conversation_deconvolution.core.config import ReconstructionConfig
from conversation_deconvolution.core.types import Utterance


class TopicTextEmbedder:
    def __init__(self, topics: dict[str, list[str]], dim: int = 64):
        self.dim = dim
        self.word_topic: dict[str, np.ndarray] = {}
        for k, topic in enumerate(sorted(topics)):
            anchor = np.zeros(dim)
            anchor[k % dim] = 1.0
            for w in topics[topic]:
                self.word_topic[w] = anchor

    def _noise_vec(self, word: str) -> np.ndarray:
        h = hashlib.md5(word.encode()).digest()
        v = np.zeros(self.dim)
        v[h[0] % self.dim] = 0.05
        return v

    def encode(self, texts):
        out = []
        for t in texts:
            v = np.zeros(self.dim)
            for w in t.lower().split():
                v += self.word_topic.get(w, self._noise_vec(w))
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.array(out)


TOPICS = {
    "cafe": ["viens", "cafe", "demain", "midi", "parfait", "reponds", "tard"],
    "rapport": ["rapport", "final", "termine", "merci", "beaucoup", "courrier"],
}


def U(i, s, a, b, text):
    return Utterance(i, s, a, b, text)


def test_empty_returns_empty():
    r = HeuristicReconstructor(TopicTextEmbedder(TOPICS), ReconstructionConfig())
    assert r.reconstruct([]) == []


def test_single_conversation():
    utts = [
        U("u1", "A", 0.0, 1.5, "tu viens au cafe demain"),
        U("u2", "B", 1.8, 3.0, "oui je viens au cafe demain"),
    ]
    convs = HeuristicReconstructor(TopicTextEmbedder(TOPICS), ReconstructionConfig()).reconstruct(utts)
    assert len(convs) == 1
    assert convs[0].id == "conversation_01"
    assert convs[0].participants == ["A", "B"]
    assert len(convs[0].utterances) == 2


def test_two_interleaved_threads_separated():
    utts = [
        U("a1", "A", 0.0, 1.8, "tu viens au cafe demain midi"),
        U("c1", "C", 0.9, 2.4, "le rapport final est termine"),
        U("a2", "A", 2.6, 4.0, "parfait pour le cafe alors"),
        U("c2", "D", 3.0, 4.5, "merci pour le rapport beaucoup"),
        U("a3", "B", 4.4, 5.8, "je reponds au cafe plus tard"),
        U("c3", "C", 4.9, 6.4, "le rapport part au courrier"),
    ]
    cfg = ReconstructionConfig()
    convs = HeuristicReconstructor(TopicTextEmbedder(TOPICS), cfg).reconstruct(utts)
    assert len(convs) == 2
    members = sorted(tuple(sorted(u.id for u in c.utterances)) for c in convs)
    assert members == [("a1", "a2", "a3"), ("c1", "c2", "c3")]
    ids = sorted(c.id for c in convs)
    assert ids == ["conversation_01", "conversation_02"]


def test_full_overlap_never_linked():
    utts = [
        U("x1", "A", 0.0, 2.0, "meme sujet identique ici"),
        U("x2", "B", 0.1, 2.0, "meme sujet identique la"),
    ]
    convs = HeuristicReconstructor(TopicTextEmbedder(TOPICS), ReconstructionConfig()).reconstruct(utts)
    assert len(convs) == 2
