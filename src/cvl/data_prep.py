from pathlib import Path
import re
import pandas as pd

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
