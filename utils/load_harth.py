import pandas as pd
import numpy as np
from pathlib import Path


def load_harth_raw(data_dir="harth"):
    data_dir = Path(data_dir)

    X_list = []
    y_list = []
    sub_list = []

    for file in sorted(data_dir.glob("S*.csv")):
        subject_id = int(file.stem.replace("S", ""))  # S006 → 6

        print(f"Loading subject {subject_id}: {file.name}")

        df = pd.read_csv(file)

        # Extract raw 6 channels
        X = df[['back_x', 'back_y', 'back_z',
                'thigh_x', 'thigh_y', 'thigh_z']].values   # (N, 6)

        # Extract label
        y = df['label'].values                            # (N,)

        # Subject ID repeated for each row
        subs = np.full(len(df), subject_id)

        # Store
        X_list.append(X)
        y_list.append(y)
        sub_list.append(subs)

    # Concatenate all subjects
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    subjects = np.concatenate(sub_list, axis=0)

    print("\nFinal Shapes:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("subjects:", subjects.shape)

    return X, y, subjects
