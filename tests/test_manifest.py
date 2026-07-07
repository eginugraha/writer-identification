from src.cvl.data_prep import scan_lines, filter_cohort, build_label_map, build_manifest

def _prep(tiny_lines):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    return kept

def test_label_map(tiny_lines):
    kept = _prep(tiny_lines)
    lm = build_label_map(kept)
    assert lm == {"0001": 0, "0002": 1}

def test_test_page_fixed_and_excluded_from_train(tiny_lines):
    kept = _prep(tiny_lines)
    m = build_manifest(kept, n_train_pages=2, seed=0)
    for w, g in m.groupby("writer"):
        test_pages = set(g[g.split == "test"]["page"])
        train_pages = set(g[g.split.isin(["train", "val"])]["page"])
        assert len(test_pages) == 1
        assert test_pages.isdisjoint(train_pages)
        assert len(train_pages) == 2  # N=2

def test_full_uses_all_non_test_pages(tiny_lines):
    kept = _prep(tiny_lines)
    m = build_manifest(kept, n_train_pages=None, seed=0)
    tp = m[(m.writer == "0001") & (m.split.isin(["train", "val"]))]["page"].nunique()
    assert tp == 5  # 6 halaman - 1 test

def test_deterministic(tiny_lines):
    kept = _prep(tiny_lines)
    a = build_manifest(kept, n_train_pages=2, seed=0)
    b = build_manifest(kept, n_train_pages=2, seed=0)
    assert a.sort_values("path").reset_index(drop=True).equals(
           b.sort_values("path").reset_index(drop=True))
