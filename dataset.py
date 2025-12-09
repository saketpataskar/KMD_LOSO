# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class UCIHARDataset(Dataset):
    """
    X: (N, T, C)
    y: (N,)
    transforms: function(X_batch) -> X_batch (applied on numpy array before tensor convert)
    """
    def __init__(self, X, y, transform=None, channels_first=False):
        self.X = X.astype("float32")
        self.y = y.astype("int64")
        self.transform = transform
        self.channels_first = channels_first

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # (T, C)
        y = self.y[idx]
        if self.transform is not None:
            x = self.transform(x)  # still numpy
        # convert to tensor; standard shapes: (C, T) for conv1d or (T, C) for LSTM
        if self.channels_first:
            x = np.transpose(x, (1, 0))  # (C, T)
        x = torch.from_numpy(x.copy()).float()
        y = torch.tensor(int(y)).long()
        return x, y

# Example simple augmentations as functions
def add_gaussian_noise(x, sigma=0.01):
    # x: (T,C)
    return x + np.random.normal(0, sigma, size=x.shape).astype("float32")

def random_crop(x, crop_ratio=0.9):
    # keep crop_ratio portion of T then pad back or stretch; simple approach: center crop
    T = x.shape[0]
    keep = int(T * crop_ratio)
    start = np.random.randint(0, T - keep + 1)
    cropped = x[start:start+keep]
    if keep < T:
        pad = np.zeros((T - keep, x.shape[1]), dtype=x.dtype)
        cropped = np.vstack([cropped, pad])
    return cropped
