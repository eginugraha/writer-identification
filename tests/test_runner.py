import pandas as pd
from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
from src.cvl.run_experiments import run_id, already_done, run_grid

def test_run_id():
    assert run_id("vit_small", 3, "pretrained", 1) == "vit_small_L3_pretrained_s1"
    assert run_id("resnet50", None, "scratch", 0) == "resnet50_Lfull_scratch_s0"

def test_run_grid_and_resume(tiny_lines, tmp_path):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    mbl = {2: build_manifest(kept, n_train_pages=2, seed=0)}
    csv = tmp_path / "results.csv"
    hp = {"num_workers": 0, "amp": False, "early_stop_patience": 1,
          "pretrained_epochs": 1, "scratch_epochs": 1, "batch_size": 8, "lr": 1e-3}
    run_grid({0: mbl}, archs=["resnet50"], levels=[2], modes=["scratch"], seeds=[0],
             results_csv=csv, ckpt_root=tmp_path / "ck", device="cpu", hp=hp)
    n1 = len(pd.read_csv(csv))
    assert n1 == 1
    assert already_done(csv, run_id("resnet50", 2, "scratch", 0))
    # jalankan lagi → tidak menambah baris (resume skip)
    run_grid({0: mbl}, archs=["resnet50"], levels=[2], modes=["scratch"], seeds=[0],
             results_csv=csv, ckpt_root=tmp_path / "ck", device="cpu", hp=hp)
    assert len(pd.read_csv(csv)) == 1
