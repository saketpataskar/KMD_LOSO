import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=8, in_channels=6, seq_len=250):
        super().__init__()

        # CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # compute output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, seq_len)
            out = self.conv(dummy)
            conv_out_channels = out.shape[1]
            conv_out_len = out.shape[2]

        # LSTM on CNN features (treat channels as features, time as sequence)
        self.lstm = nn.LSTM(
            input_size=conv_out_channels,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # Final classifier
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, channels, time)
        cnn_out = self.conv(x)                    # (batch, C, T)
        lstm_in = cnn_out.permute(0, 2, 1)        # (batch, T, C)
        lstm_out, _ = self.lstm(lstm_in)          # (batch, T, hidden)
        last_step = lstm_out[:, -1, :]            # use final LSTM output
        return self.fc(last_step)


def train_cnn_lstm(X, y, epochs=5, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = CNN_LSTM(num_classes=len(torch.unique(y)),
                     in_channels=X.shape[1],
                     seq_len=X.shape[2]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Training CNN-LSTM on:", device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(ds):.4f}")

    print("CNN-LSTM training complete.")
    return model
