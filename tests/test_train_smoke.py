import torch
from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
from src.cvl.train import RunConfig, train_one_run
from src.cvl.scenarios import Scenario, SCENARIOS

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


def _hp():
    return {"val_frac": 0.1, "num_workers": 0, "amp": False, "early_stop_patience": 1}


def _manifest(tiny_lines):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    return build_manifest(kept, n_train_pages=2, seed=0)


def test_train_dengan_scenario_default(tiny_lines, tmp_path):
    m = _manifest(tiny_lines)
    rc = RunConfig(arch="resnet50", level=2, mode="scratch", seed=0,
                   epochs=1, lr=1e-3, batch_size=8)
    out = train_one_run(m, rc, tmp_path, device="cpu", hp=_hp(), scenario=Scenario())
    assert (tmp_path / "best.pt").exists() and out["epochs_ran"] >= 1


def test_train_dengan_label_smoothing_dan_drop_path(tiny_lines, tmp_path):
    m = _manifest(tiny_lines)
    rc = RunConfig(arch="convnext_tiny", level=2, mode="scratch", seed=0,
                   epochs=1, lr=1e-3, batch_size=8)
    out = train_one_run(m, rc, tmp_path, device="cpu", hp=_hp(),
                        scenario=SCENARIOS["FT2"])
    assert out["epochs_ran"] >= 1


def test_train_dengan_freeze_strategy(tiny_lines, tmp_path):
    m = _manifest(tiny_lines)
    rc = RunConfig(arch="convnext_tiny", level=2, mode="pretrained", seed=0,
                   epochs=1, lr=1e-4, batch_size=8)
    out = train_one_run(m, rc, tmp_path, device="cpu", hp=_hp(),
                        scenario=SCENARIOS["FT3"])
    assert out["epochs_ran"] >= 1


def test_train_dengan_arcface(tiny_lines, tmp_path):
    m = _manifest(tiny_lines)
    rc = RunConfig(arch="resnet50", level=2, mode="scratch", seed=0,
                   epochs=1, lr=1e-3, batch_size=8)
    out = train_one_run(m, rc, tmp_path, device="cpu", hp=_hp(),
                        scenario=SCENARIOS["FT4"])
    assert (tmp_path / "best.pt").exists() and out["epochs_ran"] >= 1


def test_jadwal_margin_arcface():
    from src.cvl.train import arcface_margin_at
    # naik linear 0 -> m_target sepanjang epoch warmup, lalu tetap
    assert arcface_margin_at(epoch=0, warmup_epochs=3, m_target=0.3) == 0.0
    assert abs(arcface_margin_at(1, 3, 0.3) - 0.1) < 1e-9
    assert abs(arcface_margin_at(2, 3, 0.3) - 0.2) < 1e-9
    assert abs(arcface_margin_at(3, 3, 0.3) - 0.3) < 1e-9
    assert arcface_margin_at(10, 3, 0.3) == 0.3
    # tanpa warmup langsung penuh
    assert arcface_margin_at(0, 0, 0.3) == 0.3
