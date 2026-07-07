from pathlib import Path
import re
import pandas as pd
import numpy as np
from .config import EXCLUDE_WRITERS, MIN_PAGES

_LINE_RE = re.compile(r"^(\d+)-(\d+)-(\d+)\.tif$", re.IGNORECASE)

def parse_line_filename(name: str) -> tuple[str, str, int]:
    m = _LINE_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"nama file baris tak valid: {name}")
    writer, page, line = m.group(1), m.group(2), int(m.group(3))
    return writer, page, line

def scan_lines(root: Path) -> pd.DataFrame:
    rows = []
    for tif in Path(root).rglob("*.tif"):
        if "lines" not in {p.name for p in tif.parents}:
            continue  # hanya file di bawah folder lines/
        try:
            writer, page, line = parse_line_filename(tif.name)
        except ValueError:
            continue
        rows.append((writer, page, line, str(tif.resolve())))
    return pd.DataFrame(rows, columns=["writer", "page", "line", "path"])

def filter_cohort(df, min_pages: int = MIN_PAGES, exclude: set = EXCLUDE_WRITERS):
    df = df[~df["writer"].isin(exclude)].copy()
    pages_per_writer = df.groupby("writer")["page"].nunique()
    keep = pages_per_writer[pages_per_writer >= min_pages].index
    dropped = sorted(set(pages_per_writer.index) - set(keep))
    kept = df[df["writer"].isin(keep)].copy()
    info = {
        "n_excluded_rule": len(dropped),
        "n_kept_writers": len(keep),
        "dropped_writers": dropped,
    }
    return kept, info

def build_label_map(df) -> dict:
    return {w: i for i, w in enumerate(sorted(df["writer"].unique()))}

def _page_sort_key(p: str):
    return (int(p) if p.isdigit() else p)

def build_manifest(df, n_train_pages, seed: int, test_pages: int = 1, val_frac: float = 0.1):
    label_map = build_label_map(df)
    rng = np.random.default_rng(seed)
    parts = []
    for writer, g in df.groupby("writer"):
        pages = sorted(g["page"].unique(), key=_page_sort_key)
        test_p = set(pages[-test_pages:])
        pool = [p for p in pages if p not in test_p]
        if n_train_pages is not None:
            chosen = list(rng.permutation(pool))[:n_train_pages]
        else:
            chosen = pool
        g = g.copy()
        g["label"] = label_map[writer]
        g["split"] = "unused"
        g.loc[g["page"].isin(test_p), "split"] = "test"
        train_mask = g["page"].isin(chosen)
        train_lines = g[train_mask].sort_values(["page", "line"])
        n_val = max(1, int(round(len(train_lines) * val_frac))) if len(train_lines) > 1 else 0
        val_idx = set(rng.permutation(train_lines.index)[:n_val].tolist())
        g.loc[train_mask, "split"] = "train"
        g.loc[g.index.isin(val_idx), "split"] = "val"
        parts.append(g[g["split"] != "unused"])
    return pd.concat(parts, ignore_index=True)
