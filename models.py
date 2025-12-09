# models.py
import torch.nn as nn
import torch.nn.functional as F

# Simple 1D-CNN for HAR (channels-first input: (batch, C, T))
class HAR_CNN(nn.Module):
    def __init__(self, in_channels=9, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, C, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # (B, 128)
        x = self.fc(x)
        return x

# LSTM-based model (input channels last: (B, T, C))
class HAR_LSTM(nn.Module):
    def __init__(self, in_channels=9, hidden=128, num_layers=1, num_classes=6, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        mult = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden * mult, num_classes)

    def forward(self, x):
        # x: (B, T, C)
        out, _ = self.lstm(x)  # (B, T, H)
        out = out[:, -1, :]    # last time step
        out = self.fc(out)
        return out

# CNN-LSTM hybrid: conv features -> LSTM -> fc
class HAR_CNN_LSTM(nn.Module):
    def __init__(self, in_channels=9, cnn_channels=64, lstm_hidden=128, num_classes=6):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, cnn_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(cnn_channels)
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # expect (B, T, C) -> convert to (B, C, T) for conv
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn(self.conv(x)))  # (B, cnn_channels, T)
        x = x.permute(0, 2, 1)             # (B, T, cnn_channels)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
