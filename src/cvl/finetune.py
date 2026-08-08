# -*- coding: utf-8 -*-
"""
finetune.py — Strategi fine-tuning (freeze + LLRD) untuk Studi 2.
=================================================================================
Dipakai lewat `Scenario.freeze_strategy` (lihat scenarios.py); train.py yang
memanggil freeze_layers/build_param_groups dan mengoper `rc.arch`. Tidak ada
env-var: strategi ditentukan oleh skenario, arsitektur oleh RunConfig.

Strategi:
  S0  baseline          : fine-tune penuh, LR seragam (perilaku grid utama)
  S1  feature-extract   : bekukan SELURUH backbone, latih head saja
  S2  selective         : bekukan stem + 2 stage pertama, LR 1e-4
  S3  selective + LLRD  : seperti S2 + layer-wise LR decay (head 1e-4, x0.7/stage)
  S4  S3 + smoothing    : S3 + label_smoothing=0.1

Strategi dinyatakan dalam *kedalaman* (berapa stage dibekukan), bukan nama
modul harfiah, supaya berlaku untuk lebih dari satu keluarga arsitektur —
lihat LAYER_MAP di bawah. Menambah arsitektur = menambah satu entri di sana;
prefiks yang tidak cocok dengan model akan menggagalkan run, bukan didiamkan.

Scheduler, early stopping, AMP, dan augmentasi tidak disentuh modul ini.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Peta lapisan per arsitektur
#
# timm menamai pohon modul tiap keluarga arsitektur berbeda-beda: ConvNeXt
# memakai stem/stages.N, Swin memakai patch_embed/layers.N. Prefiks harfiah
# milik ConvNeXt yang dipakai pada Swin tidak cocok dengan satu parameter pun,
# sehingga freeze membekukan nol parameter dan LLRD runtuh jadi satu grup —
# tanpa error, jadi run-nya tetap "berhasil" dengan strategi yang sebenarnya
# tidak aktif. Karena itu strategi didefinisikan atas *kedalaman* (berapa stage
# dibekukan dari lapisan awal), dan nama konkretnya diambil dari peta ini.
# ---------------------------------------------------------------------------
LAYER_MAP = {
    "convnext_tiny": dict(stem="stem",
                          stages=("stages.0", "stages.1", "stages.2", "stages.3")),
    "swin_tiny": dict(stem="patch_embed",
                      stages=("layers.0", "layers.1", "layers.2", "layers.3")),
}
HEAD_PREFIX = "head"

# ---------------------------------------------------------------------------
# Definisi strategi
#   n_freeze  : jumlah stage yang dibekukan dari lapisan awal; stem ikut beku
#               bila > 0. 0 = tidak membekukan apa pun.
#   base_lr   : None = pakai rc.lr apa adanya (S0/S1); angka = override (S2-S4)
#   llrd_decay: faktor pengali LR per level ke arah lapisan awal (None = seragam)
# ---------------------------------------------------------------------------
STRATEGIES = {
    "S0": dict(n_freeze=0, base_lr=None, llrd_decay=None, label_smoothing=0.0),
    "S1": dict(n_freeze=4, base_lr=None, llrd_decay=None, label_smoothing=0.0),
    "S2": dict(n_freeze=2, base_lr=1e-4, llrd_decay=None, label_smoothing=0.0),
    "S3": dict(n_freeze=2, base_lr=1e-4, llrd_decay=0.7,  label_smoothing=0.0),
    "S4": dict(n_freeze=2, base_lr=1e-4, llrd_decay=0.7,  label_smoothing=0.1),
}


def _peta(arch: str) -> dict:
    if arch not in LAYER_MAP:
        raise ValueError(
            f"arsitektur '{arch}' belum ada di LAYER_MAP — tambahkan nama stem "
            f"dan stage-nya dulu. Terdaftar: {sorted(LAYER_MAP)}")
    return LAYER_MAP[arch]


def freeze_prefixes(arch: str, strategy: str = "S0") -> tuple:
    """Prefiks parameter yang dibekukan strategi ini pada arsitektur ini."""
    n = STRATEGIES[strategy]["n_freeze"]
    if n == 0:
        return ()
    peta = _peta(arch)
    return (peta["stem"],) + tuple(peta["stages"][:n])


def llrd_order(arch: str) -> tuple:
    """Level LLRD dari kepala ke lapisan awal.

    head mendapat base_lr, tiap langkah ke bawah dikali llrd_decay.
    """
    peta = _peta(arch)
    return (HEAD_PREFIX,) + tuple(reversed(peta["stages"])) + (peta["stem"],)


def _cek_cocok(model, prefiks, arch: str, konteks: str) -> None:
    """Gagal keras kalau ada prefiks yang tidak cocok dengan parameter mana pun.

    Tanpa ini, arsitektur yang salah peta berakhir sebagai run yang selesai
    normal dengan strategi mati — kegagalan paling mahal karena tidak terlihat.
    """
    nama = [n for n, _ in model.named_parameters()]
    hilang = [p for p in prefiks if not any(n.startswith(p) for n in nama)]
    if hilang:
        raise ValueError(
            f"{konteks}: prefiks {hilang} tidak cocok dengan parameter mana pun "
            f"pada model '{arch}'. Peta lapisan salah — periksa LAYER_MAP.")


def _prefix_of(param_name: str, urutan) -> str:
    """Kembalikan level LLRD untuk sebuah nama parameter."""
    for pfx in urutan:
        if param_name.startswith(pfx):
            return pfx
    # norm / norm_pre / lain-lain: perlakukan seperti head (LR penuh)
    return HEAD_PREFIX


def freeze_layers(model, strategy: str = "S0", arch: str = None) -> None:
    """Set requires_grad sesuai strategi. `arch` wajib — lihat _cek_cocok."""
    prefiks = freeze_prefixes(arch, strategy)
    if prefiks:
        _cek_cocok(model, prefiks, arch, f"freeze_layers({strategy})")
    for name, p in model.named_parameters():
        p.requires_grad = not name.startswith(prefiks) if prefiks else True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[finetune] {strategy}/{arch}: trainable {n_train/1e6:.1f}M / {n_total/1e6:.1f}M param")


def build_param_groups(model, strategy: str = "S0", base_lr: float = 3e-4,
                       arch: str = None):
    """
    Kembalikan param_groups untuk AdamW.
      - S0/S1: satu grup, LR = base_lr (perilaku baseline).
      - S2   : satu grup, LR = 1e-4.
      - S3/S4: grup per level dengan LLRD (head 1e-4; stage terdalam x0.7; dst).
    Hanya parameter requires_grad=True yang dimasukkan.
    """
    cfg = STRATEGIES[strategy]
    lr0 = cfg["base_lr"] if cfg["base_lr"] is not None else base_lr

    if cfg["llrd_decay"] is None:
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params, "lr": lr0}]

    urutan = llrd_order(arch)
    # Kepala boleh saja bernama lain (fallback _prefix_of menanganinya), tapi
    # stage dan stem wajib cocok — kalau tidak, LLRD runtuh jadi satu grup.
    _cek_cocok(model, urutan[1:], arch, f"build_param_groups({strategy})")

    decay = cfg["llrd_decay"]
    lr_of = {pfx: lr0 * (decay ** i) for i, pfx in enumerate(urutan)}
    buckets = {pfx: [] for pfx in urutan}
    for name, p in model.named_parameters():
        if p.requires_grad:
            buckets[_prefix_of(name, urutan)].append(p)

    groups = [{"params": ps, "lr": lr_of[pfx]} for pfx, ps in buckets.items() if ps]
    print("[finetune] LLRD LR:", {pfx: f"{lr_of[pfx]:.1e}" for pfx, ps in buckets.items() if ps})
    return groups


# ---------------------------------------------------------------------------
# Uji cepat tanpa data:  python -m src.cvl.finetune
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import timm, torch
    from .config import ALL_ARCHITECTURES

    for arch in LAYER_MAP:
        print(f"=== {arch} ({ALL_ARCHITECTURES[arch]}) ===")
        for strat in STRATEGIES:
            model = timm.create_model(ALL_ARCHITECTURES[arch], pretrained=False,
                                      num_classes=308)
            freeze_layers(model, strat, arch=arch)
            groups = build_param_groups(model, strat, base_lr=3e-4, arch=arch)
            opt = torch.optim.AdamW(groups, weight_decay=0.05)
            x = torch.randn(2, 3, 224, 224)
            loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([1, 2]))
            loss.backward(); opt.step()
            print(f"  {strat}: OK ({len(groups)} param group)\n")
    print("Semua strategi lolos smoke-test di semua arsitektur.")
