import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class CNN_Only(nn.Module):
    def __init__(self, num_classes=8, in_channels=6, in_length=250):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # compute flattened size dynamically using a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, in_length)
            conv_out = self.conv_layers(dummy)
            flat_size = conv_out.numel()  # channels * length

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


def train_cnn(X, y, epochs=10, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CNN_Only(num_classes=len(torch.unique(y_tensor)),
                      in_channels=X.shape[1],
                      in_length=X.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Training CNN on device:", device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

    print("CNN training complete.")
    return model
