import pandas as pd
from pathlib import Path

data_dir = Path("harth")
file_path = data_dir / "S006.csv"   # CSV, NOT Excel

print("Reading:", file_path.resolve())

df = pd.read_csv(file_path)   # <-- THIS IS THE FIX

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nShape:", df.shape)
