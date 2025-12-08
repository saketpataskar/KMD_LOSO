import numpy as np

def filter_valid_labels(X, y, subjects):
    """
    Keeps labels 1–12 and removes all transition labels.
    """
    valid_mask = (y >= 1) & (y <= 12)

    X = X[valid_mask]
    y = y[valid_mask]
    subjects = subjects[valid_mask]

    # Shift labels from 1–8 → 0–7
    y = y - 1

    print("After removing invalid labels:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("subjects:", subjects.shape)
    print("Unique labels:", np.unique(y))

    return X, y, subjects
