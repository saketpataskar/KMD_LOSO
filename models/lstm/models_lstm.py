import torch
import torch.nn as nn

class HARTH_LSTM(nn.Module):
    def __init__(self, num_classes=8, input_dim=6, hidden_dim=128, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, 6, 250) → we need (batch, 250, 6)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        last_timestep = lstm_out[:, -1, :]  # shape: (batch, hidden_dim)

        out = self.fc(last_timestep)
        return out
