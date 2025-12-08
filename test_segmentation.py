from utils.load_harth import load_harth_raw
from utils.preprocess_harth import filter_valid_labels
from utils.segment_harth import segment_windows

# load raw
X, y, subs = load_harth_raw("harth")

# clean labels
X, y, subs = filter_valid_labels(X, y, subs)

# 5-second segmentation
Xw, yw, subw = segment_windows(X, y, subs, window_size=250, step_size=250)

print("\nSegmentation complete.")
