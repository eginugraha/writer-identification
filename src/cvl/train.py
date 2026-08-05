import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import LineDataset
from .models import build_model, set_arcface_margin
from .finetune import freeze_layers, build_param_groups
from .scenarios import Scenario

@dataclass
class RunConfig:
    arch: str
    level: object          # int | None
    mode: str              # "pretrained" | "scratch"
    seed: int
    epochs: int
    lr: float = 3e-4
    batch_size: int = 64
    weight_decay: float = 0.05

def _seed_all(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)

def _num_classes(manifest) -> int:
    return int(manifest["label"].max()) + 1

def arcface_margin_at(epoch: int, warmup_epochs: int, m_target: float) -> float:
    """Margin ArcFace dinaikkan linear 0 -> m_target sepanjang epoch warmup.

    Tanpa ini ArcFace sering gagal konvergen di epoch awal karena head-nya
    diinisialisasi acak sementara margin sudah penuh.
    """
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return m_target
    return m_target * epoch / warmup_epochs

def train_one_run(manifest, rc: RunConfig, out_dir, device, hp: dict,
                  scenario: Scenario | None = None) -> dict:
    sc = scenario or Scenario()
    _seed_all(rc.seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = LineDataset(manifest[manifest.split == "train"], train=True,
                           geometry=sc.geometry, aug=sc.aug)
    val_ds = LineDataset(manifest[manifest.split == "val"], train=False,
                         geometry=sc.geometry, aug=sc.aug)
    nw = hp.get("num_workers", 0)
    tl = DataLoader(train_ds, batch_size=rc.batch_size, shuffle=True, num_workers=nw)
    vl = DataLoader(val_ds, batch_size=rc.batch_size, shuffle=False, num_workers=nw)
    model = build_model(rc.arch, _num_classes(manifest),
                        pretrained=(rc.mode == "pretrained"),
                        drop_path=sc.drop_path, head=sc.head).to(device)
    if sc.freeze_strategy is not None:
        freeze_layers(model, sc.freeze_strategy)
        opt = torch.optim.AdamW(
            build_param_groups(model, sc.freeze_strategy, base_lr=rc.lr),
            weight_decay=rc.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=rc.lr,
                                weight_decay=rc.weight_decay)
    # Warmup LR linear beberapa epoch lalu cosine annealing. Tanpa warmup,
    # ConvNeXt/Swin sering divergen di epoch awal lalu kolaps ke 1 kelas.
    warmup_epochs = min(int(hp.get("warmup_epochs", 3)), max(0, rc.epochs - 1))
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, rc.epochs - warmup_epochs))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [warmup, cosine], milestones=[warmup_epochs])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, rc.epochs))
    crit = torch.nn.CrossEntropyLoss(label_smoothing=sc.label_smoothing)
    use_amp = hp.get("amp", False) and device != "cpu"
    amp_device = "cuda" if device != "cpu" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)
    best_acc, best_state, patience, bad = -1.0, None, hp.get("early_stop_patience", 8), 0
    t0, epochs_ran = time.time(), 0
    lvl = "full" if rc.level is None else str(rc.level)
    tag = f"{rc.arch}_L{lvl}_{rc.mode}_s{rc.seed}"
    for epoch in range(rc.epochs):
        set_arcface_margin(model, arcface_margin_at(epoch, warmup_epochs, 0.3))
        model.train()
        loss_sum = n_seen = 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast(amp_device, enabled=use_amp):
                logits = model(x, y) if sc.head == "arcface" else model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            loss_sum += loss.item() * len(x); n_seen += len(x)
        sched.step(); epochs_ran += 1
        # validasi
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item(); total += len(y)
        acc = correct / max(1, total)
        improved = acc > best_acc
        if improved:
            best_acc, best_state, bad = acc, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        train_loss = loss_sum / max(1, n_seen)
        print(f"  [{tag}] epoch {epoch + 1}/{rc.epochs} "
              f"loss={train_loss:.4f} val_acc={acc:.4f} "
              f"best={best_acc:.4f}{' *' if improved else ''} "
              f"patience={bad}/{patience} elapsed={time.time() - t0:.0f}s",
              flush=True)
        if not improved and bad >= patience:
            print(f"  [{tag}] early stop @ epoch {epoch + 1} (best_val_acc={best_acc:.4f})", flush=True)
            break
    torch.save(best_state or model.state_dict(), out_dir / "best.pt")
    return {"best_val_acc": float(best_acc), "train_time_s": time.time() - t0, "epochs_ran": epochs_ran}
