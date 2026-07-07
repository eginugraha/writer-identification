from src.cvl.data_prep import scan_lines

def test_scan_counts(tiny_lines):
    df = scan_lines(tiny_lines)
    assert len(df) == (6 + 5 + 3) * 4
    assert set(df["writer"].unique()) == {"0001", "0002", "0003"}
    assert df.loc[df.writer == "0001", "page"].nunique() == 6
