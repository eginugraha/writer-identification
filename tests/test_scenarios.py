import dataclasses
import pytest
from src.cvl.scenarios import Scenario, SCENARIOS


def test_default_scenario_adalah_perilaku_sekarang():
    s = Scenario()
    assert s.geometry == "center"
    assert s.aug == "baseline"
    assert s.drop_path == 0.0
    assert s.label_smoothing == 0.0
    assert s.freeze_strategy is None
    assert s.head == "linear"
    assert s.eval_crops == 1


def test_scenario_beku():
    s = Scenario()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.aug = "strong"


def test_registry_lengkap():
    assert set(SCENARIOS) == {"FT0", "FT1", "FT2", "FT3", "FT4", "FT5", "FT6",
                              "FT7", "AUG"}
    assert SCENARIOS["FT0"] == Scenario()


def test_tiap_skenario_mengubah_satu_mekanisme():
    """FT1-FT4 dan AUG masing-masing berbeda dari baseline pada satu sumbu.

    FT1 dikecualikan dari hitungan satu-sumbu: geometry dan eval_crops adalah
    satu mekanisme yang sama (cakupan baris), diterapkan di sisi latih dan uji.
    """
    base = dataclasses.asdict(Scenario())
    diff_count = {}
    for name, sc in SCENARIOS.items():
        d = dataclasses.asdict(sc)
        diff_count[name] = {k for k in d if d[k] != base[k]}
    assert diff_count["FT0"] == set()
    assert diff_count["FT1"] == {"geometry", "eval_crops"}
    assert diff_count["FT2"] == {"drop_path", "label_smoothing"}
    assert diff_count["FT3"] == {"freeze_strategy"}
    assert diff_count["FT4"] == {"head"}
    assert diff_count["FT5"] == {"eval_crops"}
    assert diff_count["FT6"] == {"geometry"}
    # FT7 dikecualikan dari hitungan satu-sumbu: ia bukan mekanisme latih,
    # melainkan protokol uji FT5 yang kepalanya disamakan dengan checkpoint
    # FT4 supaya bobotnya bisa dimuat sama sekali.
    assert diff_count["FT7"] == {"head", "eval_crops"}
    assert diff_count["AUG"] == {"aug"}


def test_nilai_skenario():
    assert SCENARIOS["FT1"].geometry == "linewindow"
    assert SCENARIOS["FT1"].eval_crops == 9
    assert SCENARIOS["FT2"].drop_path == 0.2
    assert SCENARIOS["FT2"].label_smoothing == 0.1
    assert SCENARIOS["FT3"].freeze_strategy == "S3"
    assert SCENARIOS["FT4"].head == "arcface"
    assert SCENARIOS["AUG"].aug == "strong"


def test_ft5_memisahkan_sisi_uji_ft1():
    """FT5 = FT1 tanpa perubahan sisi latih.

    FT1 mengubah geometri latih *dan* protokol uji sekaligus, jadi +14,2 poin
    top1_page-nya tidak bisa dibagi antara keduanya. FT5 memakai protokol uji
    FT1 (9 jendela dirata-rata) di atas pipeline latih FT0 apa adanya, sehingga
    selisih FT5-FT0 adalah efek murni test-time ensemble dan FT1-FT5 sisanya.
    """
    assert SCENARIOS["FT5"].eval_crops == SCENARIOS["FT1"].eval_crops
    assert SCENARIOS["FT5"].geometry == SCENARIOS["FT0"].geometry
    assert dataclasses.replace(SCENARIOS["FT5"], eval_crops=1) == SCENARIOS["FT0"]


def test_freeze_strategy_dikenal_finetune():
    from src.cvl.finetune import STRATEGIES
    for sc in SCENARIOS.values():
        if sc.freeze_strategy is not None:
            assert sc.freeze_strategy in STRATEGIES
