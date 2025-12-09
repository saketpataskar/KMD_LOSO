import numpy as np

data = np.load("ucihar_raw_merged.npz")

X = data["X"]
y = data["y"]
subjects = data["subjects"]
idx_map = data["idx_map"]

print(X.shape)
print(y.shape)
print(subjects.shape)
print(idx_map.shape)
