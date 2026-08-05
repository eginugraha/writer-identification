import pandas as pd
import pytest
from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
from src.cvl.run_experiments import LR_OVERRIDES
from src.cvl.run_scenarios import (
    scenario_run_id, run_scenario_grid, hp_skenario, cek_manifest,
)


def test_scenario_run_id():
    assert scenario_run_id("FT1", 3) == "FT1_s3"
    assert scenario_run_id("AUG", 0) == "AUG_s0"


def test_hp_skenario_menerapkan_override_lr():
    # nilai override diambil dari LR_OVERRIDES (sumber kebenaran), bukan
    # literal hardcoded, supaya test ikut berubah kalau override-nya berubah
    hp = hp_skenario({"lr": 3e-4}, "convnext_tiny")
    assert hp["lr"] == LR_OVERRIDES[("convnext_tiny", "pretrained")]


def test_hp_skenario_arsitektur_tanpa_override_tidak_berubah():
    hp = hp_skenario({"lr": 3e-4}, "resnet50")
    assert hp["lr"] == 3e-4


def test_hp_skenario_tidak_mengubah_dict_input():
    asli = {"lr": 3e-4}
    hp_skenario(asli, "convnext_tiny")
    assert asli["lr"] == 3e-4


def test_cek_manifest_gagal_cepat_untuk_seed_hilang(tmp_path):
    with pytest.raises(SystemExit, match=r"seed \[3\]"):
        cek_manifest(tmp_path, seeds=[3], level=1)


def test_run_scenario_grid_dan_resume(tiny_lines, tmp_path):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    csv = tmp_path / "results-finetune.csv"
    hp = {"num_workers": 0, "amp": False, "early_stop_patience": 1,
          "pretrained_epochs": 1, "scratch_epochs": 1, "batch_size": 8, "lr": 1e-3}
    run_scenario_grid(m, names=["FT2"], seeds=[0], results_csv=csv,
                      ckpt_root=tmp_path / "ck", device="cpu", hp=hp,
                      arch="convnext_tiny", level=2)
    d = pd.read_csv(csv)
    assert len(d) == 1
    assert d.iloc[0]["scenario"] == "FT2"
    assert d.iloc[0]["run_id"] == "FT2_s0"
    assert d.iloc[0]["gpu_name"] == "cpu"
    # jalankan lagi -> resume, tidak menambah baris
    run_scenario_grid(m, names=["FT2"], seeds=[0], results_csv=csv,
                      ckpt_root=tmp_path / "ck", device="cpu", hp=hp,
                      arch="convnext_tiny", level=2)
    assert len(pd.read_csv(csv)) == 1


def test_ft0_dilewati(tiny_lines, tmp_path):
    """FT0 disalin dari results-pretrained.csv, tidak pernah dijalankan."""
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    csv = tmp_path / "results-finetune.csv"
    hp = {"num_workers": 0, "amp": False, "early_stop_patience": 1,
          "pretrained_epochs": 1, "scratch_epochs": 1, "batch_size": 8, "lr": 1e-3}
    run_scenario_grid(m, names=["FT0"], seeds=[0], results_csv=csv,
                      ckpt_root=tmp_path / "ck", device="cpu", hp=hp,
                      arch="convnext_tiny", level=2)
    assert not csv.exists()
