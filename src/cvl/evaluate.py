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

def evaluate_checkpoint(ckpt_path, manifest, arch, device, batch_size: int = 64) -> dict:
    test = manifest[manifest.split == "test"].reset_index(drop=True)
    model = build_model(arch, _num_classes(manifest), pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    ds = LineDataset(test, train=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, feats = [], []
    t0, n_img = time.time(), 0
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device); n_img += len(x)
            probs.append(torch.softmax(model(x), dim=1).cpu().numpy())
            feats.append(forward_features(model, x).cpu().numpy())
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
