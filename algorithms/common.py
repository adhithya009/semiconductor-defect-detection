"""Shared data loading, training loop and evaluation so every algorithm is compared fairly.

Same lot-aware 80/10/10 split, same class-weighted loss, same optimiser, same
epoch budget and the same metric (macro-F1) for every model.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
DATASET, RESULTS, CHECKPOINTS = ROOT / "data" / os.environ.get("WAFER_DATA", "full_dataset.npz"), ROOT / "results", ROOT / "checkpoints"
SEED, BATCH_SIZE = 42, int(os.environ.get("BATCH_SIZE", 256))
# Improvement switches (all off by default so the baseline stays reproducible):
#   V2=1  per-sample augmentation + 8-way test-time augmentation + saved probabilities for ensembling
#   TAG   suffix for result / checkpoint names, e.g. TAG=_v2
V2, TAG = os.environ.get("V2") == "1", os.environ.get("TAG", "")


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def grouped_splits(labels, groups):
    """~80/10/10 split with no wafer lot shared between train, validation and test."""
    outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    train_val, test = next(outer.split(np.zeros(len(labels)), labels, groups))
    inner = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED + 1)
    tr, va = next(inner.split(np.zeros(len(train_val)), labels[train_val], groups[train_val]))
    return train_val[tr], train_val[va], test


def augment(x):
    """Data augmentation: random rotation/reflection (defect class is invariant to these)."""
    x = torch.rot90(x, random.randrange(4), dims=(2, 3))
    if random.random() < 0.5: x = torch.flip(x, dims=(2,))
    if random.random() < 0.5: x = torch.flip(x, dims=(3,))
    return x


def augment_per_sample(x):
    """Independent random dihedral transform (rotation x reflection) for every wafer in the batch."""
    k = torch.randint(0, 4, (len(x),), device=x.device)
    flip = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    out = x.clone()
    for r in (1, 2, 3):
        m = k == r
        if m.any(): out[m] = torch.rot90(x[m], r, dims=(2, 3))
    return out


def dihedral(x, i):
    """i in 0..7: the 8 symmetries of the square."""
    return torch.rot90(x.flip(3) if i >= 4 else x, i % 4, dims=(2, 3))


def load_data(device):
    data = np.load(DATASET)
    # Values are {0, .5, 1}; store as uint8 (0,1,2) on the GPU to keep memory small.
    X = data["X"]
    maps = torch.from_numpy(X if X.dtype == np.uint8 else (X * 2).round().astype(np.uint8)).to(device)
    labels = torch.from_numpy(data["y"]).to(device)
    splits = grouped_splits(data["y"], data["groups"])
    return maps, labels, splits, [str(c) for c in data["classes"]]


def batches(maps, labels, indices, shuffle, device):
    indices = indices[torch.randperm(len(indices), device=device)] if shuffle else indices
    for start in range(0, len(indices), BATCH_SIZE):
        idx = indices[start:start + BATCH_SIZE]
        yield (maps[idx].float().div_(2).unsqueeze(1)), labels[idx]


@torch.inference_mode()
def evaluate(model, maps, labels, indices, criterion, device, prepare, tta=False, probs_out=None):
    model.eval(); loss, actual, predicted, probs = 0.0, [], [], []
    for x, y in batches(maps, labels, indices, False, device):
        with torch.autocast(device.type, dtype=torch.bfloat16):
            if tta:  # average class probabilities over the 8 dihedral views
                p = sum(model(prepare(dihedral(x, i))).float().softmax(1) for i in range(8)) / 8
                out = p.clamp_min(1e-8).log()
            else:
                out = model(prepare(x)).float()
        loss += criterion(out.float(), y).item() * len(y)
        actual.append(y.cpu()); predicted.append(out.argmax(1).cpu()); probs.append(out.float().softmax(1).cpu())
    if probs_out is not None: probs_out.append(torch.cat(probs).numpy())
    actual, predicted = torch.cat(actual).numpy(), torch.cat(predicted).numpy()
    p, r, f1, _ = precision_recall_fscore_support(actual, predicted, average="macro", zero_division=0)
    return {"loss": loss / len(actual), "accuracy": accuracy_score(actual, predicted),
            "balanced_accuracy": balanced_accuracy_score(actual, predicted),
            "macro_precision": p, "macro_recall": r, "macro_f1": f1}, actual, predicted


def run(name, build_model, epochs=20, lr=1e-3, weight_decay=1e-4, patience=5, prepare=lambda x: x,
        pretrain=None, description=""):
    """Train `build_model(num_classes)`; select the best epoch on validation macro-F1; test once.

    prepare:  maps a (B,1,32,32) batch to the model's input (e.g. resize / 3-channel).
    pretrain: optional callable(model, maps, labels, train_idx, device) for unsupervised pre-training.
    """
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else epochs
    patience = int(os.environ.get("PATIENCE", patience))
    name += TAG
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maps, labels, (train_idx, val_idx, test_idx), classes = load_data(device)
    train_idx, val_idx, test_idx = (torch.from_numpy(i).to(device) for i in (train_idx, val_idx, test_idx))
    counts = torch.bincount(labels[train_idx], minlength=len(classes)).float()
    weights = (counts.max() / counts).sqrt(); weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    model = build_model(len(classes)).to(device).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] {n_params:,} params | train {len(train_idx):,} val {len(val_idx):,} test {len(test_idx):,}")
    if pretrain: pretrain(model, maps, labels, train_idx, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epochs * -(-len(train_idx) // BATCH_SIZE))
    best_f1, stale, history, start = -1.0, 0, [], time.time()
    CHECKPOINTS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    ckpt = CHECKPOINTS / f"{name}.pt"
    for epoch in range(1, epochs + 1):
        model.train(); total, seen = 0.0, 0
        for x, y in batches(maps, labels, train_idx, True, device):
            with torch.autocast(device.type, dtype=torch.bfloat16):
                loss = criterion(model(prepare((augment_per_sample if V2 else augment)(x))).float(), y)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # guards against exploding gradients
            optimizer.step(); scheduler.step()
            total += loss.item() * len(y); seen += len(y)
        val, _, _ = evaluate(model, maps, labels, val_idx, criterion, device, prepare)
        history.append({"epoch": epoch, "train_loss": total / seen, **{f"val_{k}": v for k, v in val.items()}})
        print(f"  epoch {epoch:02d} | train loss {total / seen:.4f} | val acc {val['accuracy']:.2%} | val macro-F1 {val['macro_f1']:.2%} | {time.time() - start:.0f}s", flush=True)
        if val["macro_f1"] > best_f1:
            best_f1, stale = val["macro_f1"], 0
            torch.save({"model_state": model.state_dict(), "classes": classes, "epoch": epoch}, ckpt)
        else:
            stale += 1
            if stale >= patience: print("  early stopping"); break
    train_seconds = time.time() - start
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    test, actual, predicted = evaluate(model, maps, labels, test_idx, criterion, device, prepare, tta=V2)
    if V2:  # save validation + test probabilities (with TTA) for ensembling / decision-threshold tuning
        saved = {}
        for split, idx in (("val", val_idx), ("test", test_idx)):
            buf = []; _, y_true, _ = evaluate(model, maps, labels, idx, criterion, device, prepare, tta=True, probs_out=buf)
            saved[f"{split}_probs"], saved[f"{split}_y"] = buf[0], y_true
        (RESULTS / "probs").mkdir(exist_ok=True); np.savez_compressed(RESULTS / "probs" / f"{name}.npz", **saved)
    p, r, f1, s = precision_recall_fscore_support(actual, predicted, labels=range(len(classes)), zero_division=0)
    report = {"name": name, "description": description, "params": n_params, "train_seconds": train_seconds,
              "best_epoch": max(history, key=lambda h: h["val_macro_f1"])["epoch"], "test": test,
              "per_class_f1": {c: float(f1[i]) for i, c in enumerate(classes)}, "history": history}
    (RESULTS / f"{name}.json").write_text(json.dumps(report, indent=2))
    print(f"[{name}] TEST acc {test['accuracy']:.2%} | balanced acc {test['balanced_accuracy']:.2%} | macro-F1 {test['macro_f1']:.2%}")
