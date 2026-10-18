"""Unit 3 - LSTM: gated memory cells, each wafer map is read as a sequence of 32 rows."""
import torch.nn as nn
from common import run


class RowLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.rnn = nn.LSTM(32, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        out, _ = self.rnn(x.squeeze(1))
        return self.head(out.mean(1))


if __name__ == "__main__":
    run("lstm", RowLSTM, lr=2e-3, description="Bidirectional 2-layer LSTM over rows")
