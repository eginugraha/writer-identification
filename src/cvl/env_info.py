"""Metadata lingkungan eksekusi, dicatat per run di results.csv.

Grid sebelumnya dijalankan tanpa mencatat model GPU sama sekali, sehingga
angka waktu latih dan throughput tidak bisa diinterpretasikan belakangan.
Kolom-kolom ini juga yang membuktikan dua server memakai kartu dan versi
library yang sama.
"""
import timm
import torch


def env_metadata(device: str) -> dict:
    """Nama GPU + versi library. `device` "cpu" -> gpu_name "cpu"."""
    if device != "cpu" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "cpu"
    return {
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
    }
