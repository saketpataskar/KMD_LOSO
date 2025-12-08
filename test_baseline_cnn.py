from utils.load_harth import load_harth_raw
from utils.preprocess_harth import filter_valid_labels
from utils.segment_harth import segment_windows
from utils.prepare_harth import normalize_windows, format_for_cnn
from models.cnn.models_cnn import train_cnn

# Load and preprocess
X, y, subs = load_harth_raw("harth")
X, y, subs = filter_valid_labels(X, y, subs)
Xw, yw, subw = segment_windows(X, y, subs, window_size=250, step_size=250)
Xw = normalize_windows(Xw)
X_cnn = format_for_cnn(Xw)

# Train CNN
model = train_cnn(X_cnn, yw, epochs=3)  # 3 epochs just to test training
