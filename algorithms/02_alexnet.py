"""Unit 2 - AlexNet-style network (5 conv + 3 FC with dropout), scaled for 32x32 inputs."""
import torch.nn as nn
from common import run


def build(num_classes):
    return nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Conv2d(64, 192, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Flatten(), nn.Dropout(0.5), nn.Linear(256 * 4 * 4, 1024), nn.ReLU(inplace=True),
        nn.Dropout(0.5), nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Linear(512, num_classes))


if __name__ == "__main__":
    run("alexnet", build, lr=5e-4, description="AlexNet-style (from scratch)")
