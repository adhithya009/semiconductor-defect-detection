"""Unit 3 - vanilla (Elman) RNN: each wafer map is read as a sequence of 32 rows."""
import torch
import torch.nn as nn
from common import run


class RowRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(32)
        self.rnn = nn.RNN(32, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))
        for name, p in self.rnn.named_parameters():  # orthogonal recurrent weights keep gradients stable over 32 steps
            if "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):  # tanh recurrence is unstable in bf16
            out, _ = self.rnn(self.norm(x.squeeze(1).float()))
            return self.head(out.mean(1))


if __name__ == "__main__":
    run("rnn", RowRNN, lr=1e-3, description="Bidirectional 2-layer RNN over rows")
