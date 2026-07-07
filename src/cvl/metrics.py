import numpy as np
from sklearn.metrics import f1_score

def aggregate_by_group(probs, groups):
    gids = np.unique(groups)
    mp = np.stack([probs[groups == g].mean(axis=0) for g in gids])
    return gids, mp

def top_k_accuracy(probs, labels, k: int) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    hits = [labels[i] in topk[i] for i in range(len(labels))]
    return float(np.mean(hits))

def macro_f1(probs, labels) -> float:
    preds = probs.argmax(axis=1)
    return float(f1_score(labels, preds, average="macro"))

def retrieval_map(features, labels):
    f = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    sim = f @ f.T
    np.fill_diagonal(sim, -np.inf)  # buang diri sendiri
    n = len(labels)
    aps, top1 = [], []
    for i in range(n):
        order = np.argsort(-sim[i])
        order = order[:-1]  # exclude self (at last position due to -inf)
        rel = (labels[order] == labels[i]).astype(int)
        if len(rel) > 0:
            top1.append(rel[0])
        if rel.sum() == 0:
            continue
        cum = np.cumsum(rel)
        precision_at_hits = cum[rel == 1] / (np.where(rel == 1)[0] + 1)
        aps.append(precision_at_hits.mean())
    return float(np.mean(aps)) if aps else 0.0, float(np.mean(top1)) if top1 else 0.0
