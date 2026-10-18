"""Unit 2 - ResNet-18 style network with residual (skip) connections, trained from scratch.
Skip connections counter the vanishing-gradient problem in deeper networks."""
import torch.nn as nn
from common import run


class BasicBlock(nn.Module):
    def __init__(self, i, o, stride):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                  nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o))
        self.skip = nn.Identity() if stride == 1 and i == o else nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        return nn.functional.relu(self.body(x) + self.skip(x))


def build(num_classes):
    layers, ch = [nn.Conv2d(1, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)], 64
    for out, stride in [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)]:
        layers.append(BasicBlock(ch, out, stride)); ch = out
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(512, num_classes))


if __name__ == "__main__":
    run("resnet", build, description="ResNet-18 style (from scratch)")
