import numpy as np
from sklearn.model_selection import train_test_split

def subject_independent_split(X, y, subjects, train_size=0.7, val_size=0.15, test_size=0.15):
    # Step 1: get unique subjects
    unique_subjects = np.unique(subjects)

    # Step 2: split subjects into train + temp
    train_subj, temp_subj = train_test_split(
        unique_subjects,
        train_size=train_size,
        random_state=42
    )

    # Step 3: split temp into validation + test
    val_ratio = val_size / (val_size + test_size)

    val_subj, test_subj = train_test_split(
        temp_subj,
        train_size=val_ratio,
        random_state=42
    )

    # Step 4: map subjects back to windows
    train_mask = np.isin(subjects, train_subj)
    val_mask   = np.isin(subjects, val_subj)
    test_mask  = np.isin(subjects, test_subj)

    return (
        X[train_mask], y[train_mask], subjects[train_mask],
        X[val_mask],   y[val_mask],   subjects[val_mask],
        X[test_mask],  y[test_mask],  subjects[test_mask]
    )
