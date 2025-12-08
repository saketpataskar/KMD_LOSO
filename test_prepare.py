from utils.load_harth import load_harth_raw
from utils.preprocess_harth import filter_valid_labels
from utils.segment_harth import segment_windows
from utils.prepare_harth import normalize_windows, format_for_cnn

# Load raw
X, y, subs = load_harth_raw("harth")

# Clean
X, y, subs = filter_valid_labels(X, y, subs)

# Segment (5-second windows)
Xw, yw, subw = segment_windows(X, y, subs, window_size=250, step_size=250)

# Normalize
Xw_norm = normalize_windows(Xw)

# Format for CNN
X_cnn = format_for_cnn(Xw_norm)

print("\nFinal dataset ready for models!")
print("X_cnn shape:", X_cnn.shape)
print("Labels:", yw.shape)
print("Subjects:", subw.shape)
