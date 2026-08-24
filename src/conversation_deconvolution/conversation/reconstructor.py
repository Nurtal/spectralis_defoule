import math

import numpy as np

from conversation_deconvolution.conversation.features import UnionFind
from conversation_deconvolution.core.config import ReconstructionConfig
from conversation_deconvolution.core.types import Conversation, Utterance


def _overlap_duration(a: Utterance, b: Utterance) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


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
        n = len(ordered)
        uf = UnionFind()
        successors: dict[int, list[tuple[float, int]]] = {}
        for i in range(n):
            candidates = []
            for j in range(i + 1, n):
                if self._speakers_conflict(ordered[i].speaker, ordered[j].speaker, spans):
                    continue
                score = self._pair_score(ordered[i], ordered[j], embeddings[i], embeddings[j])
                if score >= self.cfg.threshold:
                    candidates.append((score, j))
            candidates.sort(key=lambda t: (-t[0], t[1]))
            successors[i] = [j for _, j in candidates[: self.cfg.max_successors]]
        for i in range(n):
            for j in successors[i]:
                uf.union(i, j)
        roots_in_order: list[int] = []
        for i in range(n):
            root = uf.find(i)
            if root not in roots_in_order:
                roots_in_order.append(root)
        conversations = []
        for rank, root in enumerate(roots_in_order, start=1):
            members = sorted(
                (u for u_idx, u in enumerate(ordered) if uf.find(u_idx) == root),
                key=lambda u: (u.start, u.end),
            )
            participants = []
            for u in members:
                if u.speaker and u.speaker not in participants:
                    participants.append(u.speaker)
            conversations.append(
                Conversation(
                    id=f"conversation_{rank:02d}",
                    participants=participants,
                    utterances=list(members),
                )
            )
        return conversations

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

    def _pair_score(self, a: Utterance, b: Utterance, ea, eb) -> float:
        overlap = _overlap_duration(a, b)
        shorter = min(a.duration, b.duration) or 1e-9
        if overlap / shorter > self.cfg.max_overlap_ratio:
            return -1.0
        raw_gap = b.start - a.end
        gap_val = raw_gap if raw_gap >= 0 else -overlap
        temporal = math.exp(-abs(gap_val) / self.cfg.tau)
        same = 1.0 if (a.speaker and b.speaker and a.speaker == b.speaker) else 0.0
        alternation = (
            1.0
            if (a.speaker and b.speaker and a.speaker != b.speaker and not same)
            else 0.0
        )
        semantic = max(0.0, min(1.0, float(np.dot(ea, eb))))
        return (
            self.cfg.w_temporal * temporal
            + self.cfg.w_alternation * alternation
            + self.cfg.w_semantic * semantic
            + self.cfg.w_same_speaker * same
        )

    @staticmethod
    def _normalize(embeddings):
        embeddings = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms
