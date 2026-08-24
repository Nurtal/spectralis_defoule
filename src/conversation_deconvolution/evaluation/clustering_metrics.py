import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def labels_from_conversations(conversations, keys: list[str]) -> np.ndarray:
    labels = np.full(len(keys), -1, dtype=int)
    lookup = {}
    for idx, c in enumerate(conversations):
        for u in c.utterances:
            lookup[u.id] = idx
    for pos, k in enumerate(keys):
        if k in lookup:
            labels[pos] = lookup[k]
    return labels


def pairwise_f1(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    n = len(true_labels)
    if n < 2:
        return 0.0
    tp = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_t = true_labels[i] == true_labels[j]
            same_p = pred_labels[i] == pred_labels[j]
            if same_t and same_p:
                tp += 1
            elif same_p:
                fp += 1
            elif same_t:
                fn += 1
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def conversation_metrics(
    true_conversations,
    pred_conversations,
    matched_keys: dict[str, str],
) -> dict:
    keys = sorted(matched_keys)
    t = labels_from_conversations(true_conversations, keys)
    p = labels_from_conversations(pred_conversations, [matched_keys[k] for k in keys])
    valid = (t >= 0) & (p >= 0)
    t, p = t[valid], p[valid]
    if len(t) == 0:
        return {"pairwise_f1": 0.0, "ari": 0.0, "nmi": 0.0}
    return {
        "pairwise_f1": pairwise_f1(t, p),
        "ari": float(adjusted_rand_score(t, p)),
        "nmi": float(normalized_mutual_info_score(t, p)),
    }
