"""Unit 2 - transfer learning: ImageNet-pretrained ResNet-18 fine-tuned on wafer maps.
Maps are upsampled to 64x64 and replicated to 3 channels to match the pretrained stem."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18
from common import run

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prepare(x):
    x = F.interpolate(x, size=64, mode="nearest").expand(-1, 3, -1, -1)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def build(num_classes):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, num_classes))
    return model


if __name__ == "__main__":
    run("transfer_resnet18", build, lr=5e-4, prepare=prepare, description="ImageNet ResNet-18, fine-tuned")
