import numpy as np


def segment_windows(X, y, subjects, window_size=250, step_size=250):
    """
    Segments HARTH raw data into fixed windows.
    Window size = 250 samples (5 seconds at 50 Hz)
    Non-overlapping windows by default.
    """
    X_windows = []
    y_windows = []
    subject_windows = []

    n_samples = len(X)

    # Slide window
    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size

        window_X = X[start:end]  # (250, 6)
        window_y = y[start:end]  # (250,)
        window_sub = subjects[start:end]  # (250,)

        # Majority vote label
        label = np.bincount(window_y).argmax()

        # Subject = subject of center sample
        subject = window_sub[window_size // 2]

        # Store window
        X_windows.append(window_X)
        y_windows.append(label)
        subject_windows.append(subject)

    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    subject_windows = np.array(subject_windows)

    print("Final segmented shapes:")
    print("X_windows:", X_windows.shape)  # (num_windows, 250, 6)
    print("y_windows:", y_windows.shape)
    print("subject_windows:", subject_windows.shape)

    return X_windows, y_windows, subject_windows
