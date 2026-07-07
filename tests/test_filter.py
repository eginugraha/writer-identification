from src.cvl.data_prep import scan_lines, filter_cohort

def test_filter_min_pages(tiny_lines):
    df = scan_lines(tiny_lines)
    kept, info = filter_cohort(df, min_pages=5, exclude=set())
    assert set(kept["writer"].unique()) == {"0001", "0002"}  # C punya 3 halaman → dibuang
    assert info["n_kept_writers"] == 2
    assert "0003" in info["dropped_writers"]

def test_filter_exclude(tiny_lines):
    df = scan_lines(tiny_lines)
    kept, info = filter_cohort(df, min_pages=5, exclude={"0001"})
    assert set(kept["writer"].unique()) == {"0002"}
