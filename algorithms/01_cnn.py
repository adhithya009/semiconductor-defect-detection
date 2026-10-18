"""Unit 1 - baseline CNN: conv/BN/ReLU/pool blocks, global pooling, dropout."""
import torch.nn as nn
from common import run


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))


def build(num_classes):
    return nn.Sequential(block(1, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(0.35), nn.Linear(128, num_classes))


if __name__ == "__main__":
    run("cnn", build, description="3-block CNN (from scratch)")
