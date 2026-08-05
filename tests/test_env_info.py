import torch
import timm
from src.cvl.env_info import env_metadata


def test_env_metadata_cpu():
    md = env_metadata("cpu")
    assert md["gpu_name"] == "cpu"
    assert md["torch_version"] == torch.__version__
    assert md["timm_version"] == timm.__version__


def test_env_metadata_keys_are_strings():
    md = env_metadata("cpu")
    assert set(md) == {"gpu_name", "torch_version", "timm_version"}
    assert all(isinstance(v, str) for v in md.values())


def test_katalog_seed_lima():
    from src.cvl.config import ALL_SEEDS
    assert ALL_SEEDS == [0, 1, 2, 3, 4]
