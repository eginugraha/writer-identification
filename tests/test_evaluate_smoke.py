import torch
from src.cvl.data_prep import scan_lines, filter_cohort, build_manifest
from src.cvl.train import RunConfig, train_one_run
from src.cvl.evaluate import evaluate_checkpoint, rata_rata_jendela
from src.cvl.scenarios import Scenario, SCENARIOS

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


def test_evaluate_multicrop(tiny_lines, tmp_path):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    sc = Scenario(geometry="linewindow", eval_crops=4)
    rc = RunConfig(arch="resnet50", level=2, mode="scratch", seed=0,
                   epochs=1, batch_size=8)
    train_one_run(m, rc, tmp_path, device="cpu",
                  hp={"num_workers": 0, "amp": False, "early_stop_patience": 1},
                  scenario=sc)
    res = evaluate_checkpoint(tmp_path / "best.pt", m, arch="resnet50",
                              device="cpu", batch_size=8, scenario=sc)
    assert 0.0 <= res["top1_page"] <= 1.0
    assert "map_line" in res


def test_evaluate_arcface(tiny_lines, tmp_path):
    df = scan_lines(tiny_lines)
    kept, _ = filter_cohort(df, min_pages=5, exclude=set())
    m = build_manifest(kept, n_train_pages=2, seed=0)
    sc = SCENARIOS["FT4"]
    rc = RunConfig(arch="resnet50", level=2, mode="scratch", seed=0,
                   epochs=1, batch_size=8)
    train_one_run(m, rc, tmp_path, device="cpu",
                  hp={"num_workers": 0, "amp": False, "early_stop_patience": 1},
                  scenario=sc)
    res = evaluate_checkpoint(tmp_path / "best.pt", m, arch="resnet50",
                              device="cpu", batch_size=8, scenario=sc)
    assert 0.0 <= res["top1_page"] <= 1.0


def test_rata_rata_jendela_softmax_dulu_baru_mean():
    """Urutan operasi menentukan hasilnya: softmax dulu baru dirata-rata,
    bukan sebaliknya. Logit dipilih asimetris dan renggang (bukan simetris
    seperti [10,0] vs [0,10] yang kebetulan memberi hasil sama untuk kedua
    urutan) supaya kedua urutan menghasilkan angka yang jelas berbeda
    (ketaksamaan Jensen) -- test yang cuma mengecek rentang [0,1] tidak akan
    pernah menangkap regresi urutan ini."""
    # b=1 baris, k=2 jendela, C=2 kelas. Jendela pertama sangat yakin ke
    # kelas 0, jendela kedua sama sekali tidak yakin.
    flat_logits = torch.tensor([[10.0, 0.0], [0.0, 0.0]])
    b, k = 1, 2

    softmax_dulu_baru_mean = torch.softmax(flat_logits, dim=1).reshape(b, k, -1).mean(dim=1)
    mean_dulu_baru_softmax = torch.softmax(flat_logits.reshape(b, k, -1).mean(dim=1), dim=1)

    # Nilai eksplisit (dihitung terpisah, bukan hasil rata-rata pembulatan):
    # softmax_dulu_baru_mean ~= [0.7500, 0.2500]
    # mean_dulu_baru_softmax ~= [0.9933, 0.0067]
    assert not torch.allclose(softmax_dulu_baru_mean, mean_dulu_baru_softmax, atol=1e-3)

    hasil = rata_rata_jendela(flat_logits, b, k)
    assert hasil.shape == (b, 2)
    assert torch.allclose(hasil, softmax_dulu_baru_mean)
