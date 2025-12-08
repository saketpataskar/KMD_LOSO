from utils.load_harth import load_harth_raw

X, y, subjects = load_harth_raw("harth")

print("Loaded HARTH raw data!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("subjects shape:", subjects.shape)
print("Unique subjects:", set(subjects))
print("Unique labels:", set(y))


