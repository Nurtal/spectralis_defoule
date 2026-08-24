import math

from conversation_deconvolution.core.types import Utterance


def gap(a: Utterance, b: Utterance) -> float:
    return max(0.0, b.start - a.end)


def alternation(a: Utterance, b: Utterance) -> float:
    if a.speaker is None or b.speaker is None:
        return 0.0
    return 1.0 if a.speaker != b.speaker else 0.0


def temporal_score(a: Utterance, b: Utterance, tau: float) -> float:
    return math.exp(-gap(a, b) / tau)


def candidate_pairs(utterances: list[Utterance], max_gap: float) -> list[tuple[int, int]]:
    ordered = sorted(range(len(utterances)), key=lambda i: (utterances[i].start, utterances[i].end))
    pairs = []
    for pos, i in enumerate(ordered):
        for j_pos in range(pos + 1, len(ordered)):
            j = ordered[j_pos]
            if utterances[j].start >= utterances[i].end + max_gap:
                break
            pairs.append((i, j))
    return pairs


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for x in self._parent:
            out.setdefault(self.find(x), []).append(x)
        return {r: sorted(members) for r, members in out.items()}
