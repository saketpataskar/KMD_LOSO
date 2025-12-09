# harth_preprocess.py
import os
import glob
import numpy as np
import pandas as pd

try:
    from scipy.signal import butter, filtfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ====== CONFIG – tailor to your setup ======
HARTH_ROOT = r"D:/DKE/KMD/KMD_LOSO/harth"  # folder with SXXX.csv files
OUT_NPZ = r"D:/DKE/KMD/KMD_LOSO/harth_windows.npz"

# windowing like HARTH paper: 1s @ 50 Hz, 50% overlap
WINDOW_SIZE = 50   # samples per window
STEP_SIZE = 25     # hop size (50% overlap)

# allowed labels (from HARTH description)
ALLOWED_LABELS = [
    1,   # walking
    2,   # running
    3,   # shuffling
    4,   # stairs up
    5,   # stairs down
    6,   # standing
    7,   # sitting
    8,   # lying
    13,  # cycling (sit)
    14,  # cycling (stand)
    130, # cycling (sit, inactive)
    140, # cycling (stand, inactive)
]

# optional low-pass filter
USE_FILTER = False
FILTER_CUTOFF = 20.0  # Hz
FS = 50.0             # sampling rate


def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Apply a low-pass Butterworth filter along time axis (axis=0)."""
    if not HAS_SCIPY:
        raise RuntimeError("scipy not installed; disable USE_FILTER or install scipy.")
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)


def segment_subject(df, subject_id, window_size, step_size):
    """
    Segment one subject's dataframe into overlapping windows.

    df columns assumed:
      ['timestamp','back_x','back_y','back_z',
       'thigh_x','thigh_y','thigh_z','label']
    """
    df = df[df["label"].isin(ALLOWED_LABELS)].reset_index(drop=True)
    if len(df) < window_size:
        return [], [], []

    sig = df[["back_x", "back_y", "back_z",
              "thigh_x", "thigh_y", "thigh_z"]].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int32)

    if USE_FILTER:
        sig = butter_lowpass_filter(sig, FILTER_CUTOFF, FS, order=4).astype(np.float32)

    xs, ys, subs = [], [], []
    n = sig.shape[0]

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        win_x = sig[start:end, :]      # (W, 6)
        win_y = labels[start:end]
        vals, counts = np.unique(win_y, return_counts=True)
        maj_label = vals[np.argmax(counts)]

        xs.append(win_x)
        ys.append(maj_label)
        subs.append(subject_id)

    return xs, ys, subs


def preprocess_harth(root_dir, out_npz,
                     window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                     train_ratio=0.7, seed=42):
    """
    Main preprocessing: read all SXXX.csv, segment into windows,
    assign a subject-wise train/test split (idx_map),
    map labels to 0..C-1, and save to npz.
    """
    csv_paths = sorted(glob.glob(os.path.join(root_dir, "S*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No SXXX.csv files found in {root_dir}")

    all_X = []
    all_y = []
    all_subj = []

    print(f"Found {len(csv_paths)} subject files.")
    for path in csv_paths:
        fname = os.path.basename(path)
        subj_id_str = fname.replace("S", "").replace(".csv", "")
        try:
            subj_id = int(subj_id_str)
        except ValueError:
            subj_id = len(all_subj)

        print(f"Loading {fname} (subject {subj_id})...")
        df = pd.read_csv(path)

        xs, ys, subs = segment_subject(df, subj_id, window_size, step_size)
        if not xs:
            print(f"  -> no windows created, skipping.")
            continue

        all_X.extend(xs)
        all_y.extend(ys)
        all_subj.extend(subs)
        print(f"  -> windows: {len(xs)}")

    if not all_X:
        raise RuntimeError("No windows created for any subject.")

    X = np.stack(all_X, axis=0)                  # (N, W, 6)
    y_orig = np.array(all_y, dtype=np.int32)     # original HARTH label codes
    subjects = np.array(all_subj, dtype=np.int32)

    # map labels to 0..C-1
    unique_labels = np.unique(y_orig)
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_to_idx[v] for v in y_orig], dtype=np.int64)

    # ---------- NEW: create idx_map for baseline ----------
    # subject-wise split: some subjects → train, others → test
    unique_subjects = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)
    n_train_subj = max(1, int(len(unique_subjects) * train_ratio))
    train_subjects = set(unique_subjects[:n_train_subj])
    test_subjects = set(unique_subjects[n_train_subj:])

    print("Train subjects:", sorted(train_subjects))
    print("Test subjects:", sorted(test_subjects))

    idx_map = np.empty(X.shape[0], dtype="<U5")  # string array: "train" / "test"
    for i, s in enumerate(subjects):
        idx_map[i] = "train" if s in train_subjects else "test"

    # ---------- Save ----------
    print("=== HARTH preprocessing done ===")
    print(f"Total windows: {X.shape[0]}")
    print(f"Window size: {X.shape[1]}  channels: {X.shape[2]}")
    print(f"Classes (original): {unique_labels.tolist()}")
    print(f"Subjects: {unique_subjects.tolist()}")

    np.savez_compressed(
        out_npz,
        X=X,
        y=y,
        subjects=subjects,
        idx_map=idx_map,          # important for baseline_engine
        label_values=unique_labels,
        window_size=window_size,
        step_size=step_size,
    )
    print(f"Saved npz to: {out_npz}")


if __name__ == "__main__":
    preprocess_harth(HARTH_ROOT, OUT_NPZ)
