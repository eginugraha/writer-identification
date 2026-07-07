from pathlib import Path
import re
import pandas as pd
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
