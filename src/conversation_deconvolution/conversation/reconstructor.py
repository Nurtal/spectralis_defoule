import math

import numpy as np

from conversation_deconvolution.core.config import ReconstructionConfig
from conversation_deconvolution.core.types import Conversation, Utterance


def _overlap_duration(a: Utterance, b: Utterance) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


class _Stream:
    def __init__(self):
        self.members: list[int] = []
        self.speakers: set[str] = set()
        self.last_end = -math.inf
        self.centroid = None


class HeuristicReconstructor:
    def __init__(self, text_embedder, config: ReconstructionConfig):
        self.embedder = text_embedder
        self.cfg = config

    def reconstruct(self, utterances: list[Utterance]) -> list[Conversation]:
        if not utterances:
            return []
        ordered = sorted(utterances, key=lambda u: (u.start, u.end))
        embeddings = self._normalize(self.embedder.encode([u.text for u in ordered]))
        spans: dict[str, list[tuple[float, float]]] = {}
        for u in ordered:
            if u.speaker:
                spans.setdefault(u.speaker, []).append((u.start, u.end))

        streams: list[_Stream] = []
        for i, u in enumerate(ordered):
            best = None
            best_score = self.cfg.threshold
            for st in streams:
                if self._conflicts(st, u.speaker, spans):
                    continue
                score = self._stream_score(st, u, embeddings[i])
                if score >= best_score:
                    best_score = score
                    best = st
            if best is None:
                best = _Stream()
                streams.append(best)
            self._assign(best, i, u, embeddings[i])

        conversations = []
        ranked = sorted(streams, key=lambda s: ordered[s.members[0]].start)
        for rank, st in enumerate(ranked, start=1):
            members = [
                ordered[i]
                for i in sorted(st.members, key=lambda j: (ordered[j].start, ordered[j].end))
            ]
            participants = list(dict.fromkeys(u.speaker for u in members if u.speaker))
            conversations.append(
                Conversation(
                    id=f"conversation_{rank:02d}",
                    participants=participants,
                    utterances=members,
                )
            )
        return conversations

    def _stream_score(self, st: _Stream, u: Utterance, e) -> float:
        gap = max(0.0, u.start - st.last_end)
        temporal = math.exp(-gap / self.cfg.tau)
        same = 1.0 if (u.speaker and u.speaker in st.speakers) else 0.0
        semantic = max(0.0, min(1.0, float(np.dot(e, st.centroid))))
        return (
            self.cfg.w_temporal * temporal
            + self.cfg.w_semantic * semantic
            + self.cfg.w_same_speaker * same
        )

    def _assign(self, st: _Stream, idx: int, u: Utterance, e) -> None:
        st.members.append(idx)
        if u.speaker:
            st.speakers.add(u.speaker)
        st.last_end = max(st.last_end, u.end)
        if st.centroid is None:
            st.centroid = e.copy()
        else:
            st.centroid = st.centroid + (e - st.centroid) / len(st.members)

    def _conflicts(
        self, st: _Stream, speaker: str | None, spans: dict[str, list[tuple[float, float]]]
    ) -> bool:
        if not speaker:
            return False
        others = st.speakers - {speaker}
        return any(self._speakers_conflict(speaker, other, spans) for other in others)

    def _speakers_conflict(
        self,
        speaker_a: str | None,
        speaker_b: str | None,
        spans: dict[str, list[tuple[float, float]]],
    ) -> bool:
        if not speaker_a or not speaker_b or speaker_a == speaker_b:
            return False
        if speaker_a not in spans or speaker_b not in spans:
            return False
        overlap = sum(
            max(0.0, min(e1, e2) - max(s1, s2))
            for s1, e1 in spans[speaker_a]
            for s2, e2 in spans[speaker_b]
        )
        dur_a = sum(e - s for s, e in spans[speaker_a])
        dur_b = sum(e - s for s, e in spans[speaker_b])
        shorter = min(dur_a, dur_b) or 1e-9
        return overlap / shorter > self.cfg.max_speaker_overlap_ratio

    @staticmethod
    def _normalize(embeddings):
        embeddings = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms
