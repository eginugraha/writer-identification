import pandas as pd
from src.cvl.report import summarize, pivot_markdown, efficiency_markdown

def _fake():
    rows = []
    for arch in ["resnet50", "vit_small"]:
        for level in [1, "full"]:
            for seed in [0, 1]:
                rows.append({"arch": arch, "level": level, "mode": "pretrained",
                             "seed": seed, "top1_page": 0.8, "map_line": 0.7,
                             "n_params": 1e7, "throughput_img_s": 100.0, "train_time_s": 60.0})
    return pd.DataFrame(rows)

def test_summarize_shape():
    s = summarize(_fake())
    assert {"arch", "level", "mode", "top1_page_mean", "top1_page_std"} <= set(s.columns)

def test_pivot_markdown_contains_arch():
    md = pivot_markdown(_fake(), metric="top1_page", mode="pretrained")
    assert "resnet50" in md and "|" in md

def test_efficiency_markdown_contains_arch():
    md = efficiency_markdown(_fake(), mode="pretrained")
    assert "resnet50" in md and "|" in md
