import numpy as np
from src.cvl.metrics import aggregate_by_group, top_k_accuracy, macro_f1, retrieval_map

def test_aggregate_mean():
    probs = np.array([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]])
    groups = np.array([0, 0, 1])
    gids, mp = aggregate_by_group(probs, groups)
    assert list(gids) == [0, 1]
    assert np.allclose(mp[0], [0.8, 0.2])

def test_top1_perfect():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = np.array([0, 1])
    assert top_k_accuracy(probs, labels, 1) == 1.0

def test_macro_f1_perfect():
    probs = np.eye(3)
    labels = np.array([0, 1, 2])
    assert macro_f1(probs, labels) == 1.0

def test_retrieval_map_separable():
    # dua kelas, fitur terpisah jelas → mAP sempurna
    feats = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    m, t1 = retrieval_map(feats, labels)
    assert m > 0.99 and t1 == 1.0
