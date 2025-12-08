import numpy as np

def normalize_windows(X):
    """
    Normalize each channel using mean and std across all windows.
    """
    # X shape: (n, 250, 6)
    mean = X.mean(axis=(0, 1))
    std = X.std(axis=(0, 1))

    X_norm = (X - mean) / (std + 1e-8)

    print("Normalization complete.")
    print("Mean per channel:", mean)
    print("Std per channel:", std)

    return X_norm


def format_for_cnn(X):
    """
    Reshape windows into (batch, channels, time) format required by CNNs.
    """
    # Input shape: (n, 250, 6)
    X_cnn = X.transpose(0, 2, 1)

    print("Reshaped to:", X_cnn.shape)
    return X_cnn
