import networkx as nx
import numpy as np
from scipy.special import expit

from conversation_deconvolution.conversation.features import candidate_pairs
from conversation_deconvolution.conversation.pair_features import pair_features
from conversation_deconvolution.conversation.trainer import load_model
from conversation_deconvolution.core.config import GraphConfig
from conversation_deconvolution.core.types import Conversation, Utterance


class GraphReconstructor:
    def __init__(self, text_embedder, config: GraphConfig):
        self.embedder = text_embedder
        self.cfg = config
        self.model = load_model(config.model_path)

    def reconstruct(self, utterances: list[Utterance]) -> list[Conversation]:
        if not utterances:
            return []
        ordered = sorted(utterances, key=lambda u: (u.start, u.end))
        embs = self._normalize(self.embedder.encode([u.text for u in ordered]))
        probs = self._edge_probabilities(ordered, embs)
        communities = self._communities(ordered, probs)
        return self._to_conversations(ordered, communities)

    def _edge_probabilities(self, ordered, embs) -> dict[tuple[int, int], float]:
        pairs = candidate_pairs(ordered, self.cfg.max_gap)
        if not pairs:
            return {}
        X = np.asarray(
            [
                pair_features(
                    ordered[i],
                    ordered[j],
                    i,
                    j,
                    float(np.dot(embs[i], embs[j])),
                    self.cfg.tau,
                )
                for i, j in pairs
            ],
            dtype=np.float64,
        )
        mean = np.asarray(self.model["scaler"]["mean"], dtype=np.float64)
        scale = np.asarray(self.model["scaler"]["scale"], dtype=np.float64)
        coef = np.asarray(self.model["coef"], dtype=np.float64)
        scores = (X - mean) / scale @ coef + float(self.model["intercept"])
        return {(i, j): float(p) for (i, j), p in zip(pairs, expit(scores))}

    def _communities(self, ordered, probs):
        G = nx.Graph()
        G.add_nodes_from(range(len(ordered)))
        for (i, j), p in probs.items():
            if p >= self.cfg.edge_threshold:
                G.add_edge(i, j, weight=p)
        return nx.community.louvain_communities(
            G, weight="weight", resolution=self.cfg.resolution, seed=self.cfg.seed
        )

    def _to_conversations(self, ordered, communities) -> list[Conversation]:
        groups = sorted(communities, key=lambda m: ordered[min(m)].start)
        conversations = []
        for rank, members in enumerate(groups, start=1):
            utts = [ordered[i] for i in sorted(members)]
            participants = list(dict.fromkeys(u.speaker for u in utts if u.speaker))
            conversations.append(
                Conversation(
                    id=f"conversation_{rank:02d}",
                    participants=participants,
                    utterances=utts,
                )
            )
        return conversations

    @staticmethod
    def _normalize(embeddings):
        embeddings = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms
