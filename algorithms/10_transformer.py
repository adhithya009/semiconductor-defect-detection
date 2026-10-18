"""Unit 3 - Vision Transformer: 4x4 patches -> 64 tokens + [CLS], multi-head self-attention encoder."""
import torch
import torch.nn as nn
from common import run


class WaferViT(nn.Module):
    def __init__(self, num_classes, dim=128, depth=6, heads=4, patch=4):
        super().__init__()
        self.embed = nn.Conv2d(1, dim, patch, patch)
        n = (32 // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        tokens = self.embed(x).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], 1) + self.pos
        return self.head(self.norm(self.encoder(tokens)[:, 0]))


if __name__ == "__main__":
    run("transformer", WaferViT, lr=5e-4, weight_decay=0.05, description="Vision Transformer (from scratch)")
