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

# -----------------------------
# InceptionTime Blocks
# -----------------------------

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(9, 19, 39), bottleneck_channels=32):
        super().__init__()

        self.use_bottleneck = in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            conv_input_channels = bottleneck_channels
        else:
            conv_input_channels = in_channels

        self.conv_list = nn.ModuleList([
            nn.Conv1d(conv_input_channels, out_channels, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        # Bottleneck
        if self.use_bottleneck:
            x_bottleneck = self.bottleneck(x)
        else:
            x_bottleneck = x

        conv_outputs = [conv(x_bottleneck) for conv in self.conv_list]
        pool_out = self.conv_pool(self.maxpool(x))

        x = torch.cat(conv_outputs + [pool_out], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x


# -----------------------------
# InceptionTime Network
# -----------------------------

class HAR_InceptionTime(nn.Module):
    def __init__(self, in_channels=9, num_classes=6, num_modules=3, out_channels=32):
        super().__init__()

        modules = []
        current_channels = in_channels

        for _ in range(num_modules):
            module = InceptionModule(
                in_channels=current_channels,
                out_channels=out_channels
            )
            modules.append(module)
            current_channels = out_channels * 4  # 3 convs + 1 pool branch

        self.inception_stack = nn.Sequential(*modules)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(current_channels, num_classes)

    def forward(self, x):
        # x: (B, C, T)
        x = self.inception_stack(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x
