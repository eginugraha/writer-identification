import torch
from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
from src.cvl.train import RunConfig, train_one_run

def test_train_smoke(tiny_lines, tmp_path):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    rc = RunConfig(arch="resnet50", level=2, mode="scratch", seed=0,
                   epochs=1, lr=1e-3, batch_size=8)
    hp = {"val_frac": 0.1, "num_workers": 0, "amp": False, "early_stop_patience": 1}
    out = train_one_run(m, rc, tmp_path, device="cpu", hp=hp)
    assert (tmp_path / "best.pt").exists()
    assert "best_val_acc" in out and out["epochs_ran"] >= 1
