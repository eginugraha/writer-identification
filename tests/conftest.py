import pytest
from PIL import Image

def _make_line(dirpath, writer, page, line, w=120, h=48):
    dirpath.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (w, h), (200, 200, 200))
    img.save(dirpath / f"{writer}-{page}-{line}.tif")

@pytest.fixture
def tiny_lines(tmp_path):
    """3 penulis: A=6 halaman, B=5 halaman, C=3 halaman (di bawah min)."""
    lines = tmp_path / "lines"
    plan = {"0001": 6, "0002": 5, "0003": 3}
    for writer, npages in plan.items():
        for p in range(1, npages + 1):
            for ln in range(4):  # 4 baris/halaman
                _make_line(lines / writer, writer, p, ln)
    return tmp_path

@pytest.fixture
def wide_line_image():
    """Citra baris dengan rasio ~12:1, meniru dimensi asli CVL (1739x137)."""
    return Image.new("RGB", (1740, 140), (200, 200, 200))
