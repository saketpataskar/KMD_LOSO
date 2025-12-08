from utils.load_harth import load_harth_raw
from utils.preprocess_harth import filter_valid_labels

# Load raw (already working)
X, y, subjects = load_harth_raw("harth")

# Filter invalid / transition labels
X_clean, y_clean, subjects_clean = filter_valid_labels(X, y, subjects)
