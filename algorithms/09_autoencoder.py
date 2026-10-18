"""Unit 3 - convolutional autoencoder: unsupervised pre-training (reconstruction), then a classifier head
on the learned encoder is fine-tuned with labels."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import augment, batches, run

PRETRAIN_EPOCHS = 5


def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class AutoencoderClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(cbr(1, 32), nn.MaxPool2d(2), cbr(32, 64), nn.MaxPool2d(2), cbr(64, 128), nn.MaxPool2d(2))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, 2), cbr(64, 64), nn.ConvTranspose2d(64, 32, 2, 2), cbr(32, 32),
                                     nn.ConvTranspose2d(32, 16, 2, 2), nn.Conv2d(16, 1, 3, padding=1))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.35), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(self.encoder(x))


def pretrain(model, maps, labels, train_idx, device):
    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train(); total, seen = 0.0, 0
        for x, _ in batches(maps, labels, train_idx, True, device):
            x = augment(x)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                loss = F.mse_loss(model.decoder(model.encoder(x)).float(), x)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += loss.item() * len(x); seen += len(x)
        print(f"  pretrain epoch {epoch} | reconstruction MSE {total / seen:.5f}", flush=True)


if __name__ == "__main__":
    run("autoencoder", AutoencoderClassifier, pretrain=pretrain, description="Conv autoencoder pre-train + fine-tune")
