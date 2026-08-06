"""Cek pra-terbang pod: GPU benar-benar terpakai, VRAM cukup, worker masuk akal.

Jalankan sekali setelah `pip install`, sebelum prep_manifests.py:

    python scripts/preflight.py

Menjawab tiga pertanyaan yang kalau salah baru ketahuan berjam-jam kemudian:
  1. Apakah build PyTorch punya kernel untuk kartu ini? (kartu generasi baru
     seperti Blackwell butuh CUDA 12.8+ dan kernel sm_120; tanpa itu prosesnya
     mati di peluncuran kernel pertama, bukan saat torch.cuda.is_available())
  2. Berapa VRAM puncak sebenarnya pada batch dari configs/default.yaml?
  3. Berapa num_workers yang pantas untuk vCPU pod ini, dan apakah /dev/shm
     cukup lapang untuk antrean prefetch-nya?
"""
import os
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import timm
    import torch
    import yaml
except ModuleNotFoundError as e:
    # Cek ini gunanya mendiagnosis pod, jadi ia tidak boleh ikut mati dengan
    # traceback yang tidak menjelaskan apa-apa.
    sys.exit(
        f"Paket '{e.name}' belum terpasang.\n"
        "Dependensi belum dipasang di environment ini. Dari root repo:\n"
        "    pip install -r requirements.txt\n"
        "Kalau ini pod kedua, pakai berkas terkunci dari pod pertama:\n"
        "    pip install -r requirements.lock.txt\n"
        "Pastikan venv-nya aktif (prompt diawali '(.venv)') lalu ulangi."
    )

# Model dengan aktivasi terbesar di katalog — patokan VRAM kasus terburuk.
ARCH_TERBERAT = "tf_efficientnetv2_s"
N_KELAS = 308


def _gb(x):
    return x / 1e9


def cek_versi():
    print("== versi ==")
    print(f"  torch {torch.__version__} | CUDA {torch.version.cuda} | timm {timm.__version__}")


def cek_gpu():
    """True bila GPU siap dipakai; False bila jatuh ke CPU."""
    print("\n== GPU ==")
    if not torch.cuda.is_available():
        print("  TIDAK ADA CUDA — run akan jalan di CPU dan makan waktu berminggu-minggu.")
        print("  Periksa driver pod dan build torch (butuh varian +cu, bukan CPU-only).")
        return False
    nama = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    arch = torch.cuda.get_arch_list()
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"  {nama} | compute capability {cc[0]}.{cc[1]} | VRAM {_gb(total):.0f} GB")
    print(f"  kernel tersedia di build ini: {arch}")
    butuh = f"sm_{cc[0]}{cc[1]}"
    if butuh not in arch:
        print(f"  !! build torch ini TIDAK memuat {butuh}.")
        print("     Kartu generasi baru butuh PyTorch dengan CUDA 12.8+. Ganti image pod")
        print("     atau pasang ulang torch sebelum melanjutkan.")
    return True


def cek_beban(pakai_gpu: bool, batch: int, amp: bool):
    """Forward + backward sungguhan — di sinilah masalah kernel muncul."""
    print(f"\n== forward+backward {ARCH_TERBERAT}, batch {batch} ==")
    dev = "cuda" if pakai_gpu else "cpu"
    if not pakai_gpu:
        batch = min(batch, 4)
        print(f"  (CPU: batch dikecilkan ke {batch} supaya cepat)")
    m = timm.create_model(ARCH_TERBERAT, pretrained=False, num_classes=N_KELAS).to(dev)
    x = torch.randn(batch, 3, 224, 224, device=dev)
    y = torch.zeros(batch, dtype=torch.long, device=dev)
    with torch.amp.autocast(dev, enabled=(amp and pakai_gpu)):
        torch.nn.functional.cross_entropy(m(x), y).backward()
    if pakai_gpu:
        torch.cuda.synchronize()
        puncak = torch.cuda.max_memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        print(f"  OK — VRAM puncak {_gb(puncak):.1f} GB dari {_gb(total):.0f} GB "
              f"({100 * puncak / total:.0f}%)")
        if puncak > 0.85 * total:
            print("  !! sangat mepet. Turunkan batch_size di configs/default.yaml.")
    else:
        print("  OK (CPU)")


def cek_cpu_dan_shm(num_workers: int, prefetch: int, batch: int):
    print("\n== CPU, RAM, shared memory ==")
    vcpu = os.cpu_count()
    saran = max(1, vcpu - 2)
    print(f"  vCPU terdeteksi: {vcpu} -> saran num_workers = {saran} "
          f"(sekarang {num_workers})")
    if num_workers != saran:
        print(f"  !! configs/default.yaml menyetel num_workers={num_workers}. "
              f"Beban ini terbatas CPU, bukan GPU — sesuaikan ke {saran}.")

    # Antrean prefetch hidup di /dev/shm; loader latih dan validasi sama-sama
    # persistent sehingga biayanya dua kali.
    per_batch = batch * 3 * 224 * 224 * 4
    butuh = 2 * num_workers * prefetch * per_batch
    print(f"  antrean prefetch: 2 loader x {num_workers} worker x {prefetch} "
          f"= {_gb(butuh):.1f} GB di shared memory")
    if Path("/dev/shm").exists():
        shm = shutil.disk_usage("/dev/shm").total
        print(f"  /dev/shm tersedia: {_gb(shm):.1f} GB")
        if shm < butuh * 1.5:
            print("  !! terlalu sempit. Gejalanya 'DataLoader worker killed by "
                  "signal: Bus error' beberapa menit setelah mulai.")
            print(f"     Turunkan prefetch_factor, atau jalankan pod dengan "
                  f"--shm-size minimal {_gb(butuh * 2):.0f}g.")
    else:
        print("  /dev/shm tidak ada (bukan Linux) — lewati")


def main():
    hp = yaml.safe_load(open("configs/default.yaml"))
    cek_versi()
    pakai_gpu = cek_gpu()
    cek_beban(pakai_gpu, hp["batch_size"], hp.get("amp", False))
    cek_cpu_dan_shm(hp.get("num_workers", 0), hp.get("prefetch_factor", 4),
                    hp["batch_size"])
    print("\nSelesai. Perbaiki setiap baris bertanda '!!' sebelum menjalankan grid.")


if __name__ == "__main__":
    main()
