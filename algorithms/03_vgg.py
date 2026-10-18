"""Unit 2 - VGG-style network: stacks of 3x3 convolutions, channels doubling per stage."""
import torch.nn as nn
from common import run


def stage(i, o, n):
    layers = []
    for k in range(n):
        layers += [nn.Conv2d(i if k == 0 else o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers, nn.MaxPool2d(2))


def build(num_classes):
    return nn.Sequential(stage(1, 64, 2), stage(64, 128, 2), stage(128, 256, 3), stage(256, 256, 3),
                         nn.Flatten(), nn.Dropout(0.5), nn.Linear(256 * 2 * 2, 512), nn.ReLU(inplace=True),
                         nn.Dropout(0.5), nn.Linear(512, num_classes))


if __name__ == "__main__":
    run("vgg", build, description="VGG-style (from scratch)")
