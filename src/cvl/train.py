import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import LineDataset
from .models import build_model

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

def train_one_run(manifest, rc: RunConfig, out_dir, device, hp: dict) -> dict:
    _seed_all(rc.seed)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = LineDataset(manifest[manifest.split == "train"], train=True)
    val_ds = LineDataset(manifest[manifest.split == "val"], train=False)
    nw = hp.get("num_workers", 0)
    tl = DataLoader(train_ds, batch_size=rc.batch_size, shuffle=True, num_workers=nw)
    vl = DataLoader(val_ds, batch_size=rc.batch_size, shuffle=False, num_workers=nw)
    model = build_model(rc.arch, _num_classes(manifest), pretrained=(rc.mode == "pretrained")).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=rc.lr, weight_decay=rc.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, rc.epochs))
    crit = torch.nn.CrossEntropyLoss()
    use_amp = hp.get("amp", False) and device != "cpu"
    amp_device = "cuda" if device != "cpu" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=use_amp)
    best_acc, best_state, patience, bad = -1.0, None, hp.get("early_stop_patience", 8), 0
    t0, epochs_ran = time.time(), 0
    for epoch in range(rc.epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast(amp_device, enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step(); epochs_ran += 1
        # validasi
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item(); total += len(y)
        acc = correct / max(1, total)
        if acc > best_acc:
            best_acc, best_state, bad = acc, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    torch.save(best_state or model.state_dict(), out_dir / "best.pt")
    return {"best_val_acc": float(best_acc), "train_time_s": time.time() - t0, "epochs_ran": epochs_ran}
