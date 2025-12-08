from utils.load_harth import load_harth_raw
from utils.preprocess_harth import filter_valid_labels
from utils.segment_harth import segment_windows
from utils.prepare_harth import normalize_windows, format_for_cnn
from utils.dataset_splits import subject_independent_split

# Load data
X, y, subs = load_harth_raw("harth")
X, y, subs = filter_valid_labels(X, y, subs)
Xw, yw, subw = segment_windows(X, y, subs, window_size=250, step_size=250)
Xw = normalize_windows(Xw)
X_cnn = format_for_cnn(Xw)

# Split
X_train, y_train, s_train, X_val, y_val, s_val, X_test, y_test, s_test = subject_independent_split(
    X_cnn, yw, subw
)

print("Train:", X_train.shape, "Subjects:", set(s_train))
print("Val:  ", X_val.shape,   "Subjects:", set(s_val))
print("Test: ", X_test.shape,  "Subjects:", set(s_test))

# CHECK SUBJECT LEAKAGE
print("\nOverlap check (should be empty):")
print("Train ∩ Val:", set(s_train) & set(s_val))
print("Train ∩ Test:", set(s_train) & set(s_test))
print("Val ∩ Test:", set(s_val) & set(s_test))
