import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import LineDataset
from .models import build_model, forward_features, count_params
from .metrics import aggregate_by_group, top_k_accuracy, macro_f1, retrieval_map

def _num_classes(manifest) -> int:
    return int(manifest["label"].max()) + 1

def rata_rata_jendela(logits, b: int, k: int):
    """Rata-ratakan prediksi K jendela per baris.

    Urutannya menentukan: softmax dulu, baru dirata-rata. Merata-ratakan logit
    lalu men-softmax menghasilkan besaran yang berbeda (ketaksamaan Jensen) dan
    akan mengubah setiap metrik multi-crop tanpa satu test pun gagal.
    """
    return torch.softmax(logits, dim=1).reshape(b, k, -1).mean(dim=1)

def evaluate_checkpoint(ckpt_path, manifest, arch, device, batch_size: int = 64,
                        scenario=None) -> dict:
    from .scenarios import Scenario
    sc = scenario or Scenario()
    test = manifest[manifest.split == "test"].reset_index(drop=True)
    model = build_model(arch, _num_classes(manifest), pretrained=False,
                        drop_path=sc.drop_path, head=sc.head).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    ds = LineDataset(test, train=False, geometry=sc.geometry, aug=sc.aug,
                     eval_crops=sc.eval_crops)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, feats = [], []
    t0, n_img = time.time(), 0
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            if x.dim() == 5:
                # [B, K, 3, H, W] -> forward semua jendela, rata-ratakan per baris
                b, k = x.shape[0], x.shape[1]
                flat = x.reshape(b * k, *x.shape[2:])
                # Satuan throughput adalah baris/detik, bukan jendela/detik --
                # jangan diganti ke flat.shape[0], nanti angkanya menipu.
                n_img += x.shape[0]
                p = rata_rata_jendela(model(flat), b, k)
                f = forward_features(model, flat).reshape(b, k, -1).mean(dim=1)
            else:
                n_img += len(x)
                p = torch.softmax(model(x), dim=1)
                f = forward_features(model, x)
            probs.append(p.cpu().numpy())
            feats.append(f.cpu().numpy())
    throughput = n_img / max(1e-6, time.time() - t0)
    probs = np.concatenate(probs); feats = np.concatenate(feats)
    labels = test["label"].to_numpy()
    page_groups = (test["writer"] + "|" + test["page"]).to_numpy()
    gids, page_probs = aggregate_by_group(probs, page_groups)
    page_labels = np.array([test[page_groups == g]["label"].iloc[0] for g in gids])
    map_line, top1_retrieval = retrieval_map(feats, labels)
    return {
        "top1_page": top_k_accuracy(page_probs, page_labels, 1),
        "top5_page": top_k_accuracy(page_probs, page_labels, min(5, page_probs.shape[1])),
        "macro_f1_page": macro_f1(page_probs, page_labels),
        "map_line": map_line,
        "top1_retrieval": top1_retrieval,
        "n_params": count_params(model),
        "throughput_img_s": throughput,
    }
