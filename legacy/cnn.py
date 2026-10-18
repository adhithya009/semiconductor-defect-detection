"""Train a generalizable CNN on every labeled wafer map.

Run ``python prep_data.py`` once, then ``python cnn.py``. A group-aware split
keeps wafer lots separate across train, validation, and test data.
"""
import csv
import json
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

DATASET, CHECKPOINT, METRICS_FILE = "full_dataset.npz", "wafer_cnn.pth", "training_metrics.csv"
SEED, EPOCHS, BATCH_SIZE, PATIENCE = 42, 25, 256, 6
LEARNING_RATE, WEIGHT_DECAY = 1e-3, 1e-4


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


class WaferDataset(Dataset):
    def __init__(self, maps, labels, indices, augment=False):
        self.maps, self.labels, self.indices, self.augment = maps, labels, indices, augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, position):
        image = torch.from_numpy(self.maps[self.indices[position]]).unsqueeze(0).float()
        if self.augment:
            # Defect type is invariant under 90-degree rotation and reflection.
            image = torch.rot90(image, random.randrange(4), dims=(1, 2))
            if random.random() < 0.5: image = torch.flip(image, dims=(1,))
            if random.random() < 0.5: image = torch.flip(image, dims=(2,))
        return image, int(self.labels[self.indices[position]])


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True), nn.MaxPool2d(2))


class WaferCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.35), nn.Linear(128, num_classes))

    def forward(self, x): return self.classifier(self.features(x))


def grouped_splits(labels, groups):
    """Return approximately 80/10/10 splits with no lot in multiple splits."""
    outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    train_val, test = next(outer.split(np.zeros(len(labels)), labels, groups))
    inner = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=SEED + 1)
    inner_train, inner_val = next(inner.split(np.zeros(len(train_val)), labels[train_val], groups[train_val]))
    return train_val[inner_train], train_val[inner_val], test


@torch.inference_mode()
def evaluate(model, loader, device, criterion):
    model.eval(); losses, actual, predicted = [], [], []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        losses.append(criterion(logits, labels).item() * len(labels))
        actual.extend(labels.cpu().numpy()); predicted.extend(logits.argmax(1).cpu().numpy())
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average="macro", zero_division=0)
    return {"loss": sum(losses) / len(actual), "accuracy": accuracy_score(actual, predicted), "balanced_accuracy": balanced_accuracy_score(actual, predicted), "macro_precision": precision, "macro_recall": recall, "macro_f1": f1, "actual": actual, "predicted": predicted}


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    data = np.load(DATASET)
    maps, labels, groups = data["X"], data["y"], data["groups"]
    classes = [str(name) for name in data["classes"]]
    train_indices, val_indices, test_indices = grouped_splits(labels, groups)
    print(f"Split sizes — train: {len(train_indices):,}, validation: {len(val_indices):,}, test: {len(test_indices):,}")
    loader_options = {"batch_size": BATCH_SIZE, "num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(WaferDataset(maps, labels, train_indices, augment=True), shuffle=True, **loader_options)
    val_loader = DataLoader(WaferDataset(maps, labels, val_indices), **loader_options)
    test_loader = DataLoader(WaferDataset(maps, labels, test_indices), **loader_options)

    # All maps are included. Inverse-square-root weights offset the dominant
    # "none" label without giving Near-full an unstable thousand-fold weight.
    counts = np.bincount(labels[train_indices], minlength=len(classes))
    class_weights = np.sqrt(counts.max() / counts); class_weights /= class_weights.mean()
    print("Class weights:", dict(zip(classes, np.round(class_weights, 2))))
    model = WaferCNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_f1, stale_epochs, history = -1.0, 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train(); running_loss, examples = 0.0, 0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True); loss = criterion(model(images), targets); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
            running_loss += loss.item() * len(targets); examples += len(targets)
        validation = evaluate(model, val_loader, device, criterion); scheduler.step(validation["macro_f1"])
        row = {"epoch": epoch, "train_loss": running_loss / examples, **{k: validation[k] for k in validation if k not in {"actual", "predicted"}}}
        history.append(row)
        print(f"Epoch {epoch:02d} | train loss {row['train_loss']:.4f} | val accuracy {row['accuracy']:.2%} | val macro F1 {row['macro_f1']:.2%}")
        if validation["macro_f1"] > best_f1:
            best_f1, stale_epochs = validation["macro_f1"], 0
            torch.save({"model_state": model.state_dict(), "classes": classes, "image_size": int(maps.shape[-1]), "epoch": epoch, "validation_macro_f1": best_f1}, CHECKPOINT)
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print("Early stopping: validation macro-F1 did not improve."); break
    with open(METRICS_FILE, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False); model.load_state_dict(checkpoint["model_state"])
    test = evaluate(model, test_loader, device, criterion)
    scores = precision_recall_fscore_support(test["actual"], test["predicted"], labels=range(len(classes)), zero_division=0)
    report = {key: value for key, value in test.items() if key not in {"actual", "predicted"}}
    report["best_epoch"] = checkpoint["epoch"]
    report["per_class"] = {name: {"precision": float(scores[0][i]), "recall": float(scores[1][i]), "f1": float(scores[2][i]), "support": int(scores[3][i])} for i, name in enumerate(classes)}
    with open("test_metrics.json", "w") as file: json.dump(report, file, indent=2)
    print("\nFinal untouched test metrics\n" + json.dumps(report, indent=2))


if __name__ == "__main__": main()
