"""Unit 2 - GoogLeNet-style network built from Inception modules (parallel 1x1/3x3/5x5/pool branches)."""
import torch
import torch.nn as nn
from common import run


def conv(i, o, k):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Inception(nn.Module):
    def __init__(self, i, c1, c3r, c3, c5r, c5, cp):
        super().__init__()
        self.b1 = conv(i, c1, 1)
        self.b3 = nn.Sequential(conv(i, c3r, 1), conv(c3r, c3, 3))
        self.b5 = nn.Sequential(conv(i, c5r, 1), conv(c5r, c5, 5))
        self.bp = nn.Sequential(nn.MaxPool2d(3, 1, 1), conv(i, cp, 1))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], 1)


def build(num_classes):
    return nn.Sequential(conv(1, 64, 3), Inception(64, 32, 32, 64, 8, 16, 16), nn.MaxPool2d(2),
                         Inception(128, 64, 64, 96, 16, 32, 32), nn.MaxPool2d(2),
                         Inception(224, 96, 96, 128, 16, 32, 32), nn.MaxPool2d(2),
                         Inception(288, 128, 96, 160, 24, 48, 48),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(384, num_classes))


if __name__ == "__main__":
    run("googlenet", build, description="GoogLeNet/Inception-style (from scratch)")
