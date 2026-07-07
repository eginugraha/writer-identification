from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
from src.cvl.train import RunConfig, train_one_run
from src.cvl.evaluate import evaluate_checkpoint

def test_evaluate_smoke(tiny_lines, tmp_path):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    rc = RunConfig(arch="resnet50", level=2, mode="scratch", seed=0, epochs=1, batch_size=8)
    train_one_run(m, rc, tmp_path, device="cpu",
                  hp={"num_workers": 0, "amp": False, "early_stop_patience": 1})
    res = evaluate_checkpoint(tmp_path / "best.pt", m, arch="resnet50", device="cpu", batch_size=8)
    for k in ["top1_page", "macro_f1_page", "map_line", "top1_retrieval", "n_params"]:
        assert k in res
    assert 0.0 <= res["top1_page"] <= 1.0
