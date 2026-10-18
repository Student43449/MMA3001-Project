"""Prepare TrackMan golf-shot data for regression and neural-network modelling.

Reads the original CSV from the repository's data files directory, skips the
measurement-units row, removes unused columns, keeps Driver shots, removes rows
with missing values, and writes the cleaned dataset. It also creates four
shuffled data partitions plus separate training and validation CSV files.

Run this script before either modelling script. Paths are resolved relative to
this file, so the current working directory does not need to be the repository
root.
"""

import pandas as pd
from pathlib import Path

# Resolve paths from this script's location, not the current working directory.
# This assumes the script is stored in the repository's "Code files" directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
data_folder = REPO_ROOT / "data files"
input_file = data_folder / "data.csv"
output_file = data_folder / "cleaned_data.csv"

# 1. Load the data, skipping the units row
df = pd.read_csv(input_file, skiprows=[1])
print(f"Rows initially loaded: {len(df)}")

# 2. Remove columns not required for the model
# These columns contain missing values and are excluded from the cleaned dataset.
columns_to_drop = [
    "Ball",
    "Dynamic Lie",
    "Impact Offset",
    "Impact Height",
]
df = df.drop(columns=columns_to_drop)

# 3. Keep only Driver shots
df = df[df["Club"] == "Driver"].copy()
print(f"Rows after filtering for Driver: {len(df)}")

# 4. Remove rows containing any missing values
df = df.dropna()
print(f"Rows after removing missing values: {len(df)}")

# 5. Save the cleaned dataset
df.to_csv(output_file, index=False)

print(f"Cleaned data saved to: {output_file}")
print(f"Final number of rows: {len(df)}")
print(f"Final number of columns: {len(df.columns)}")


# 6. Divide the cleaned dataset into four approximately equal, shuffled datasets.
# A fixed random seed makes the split reproducible whenever this script is run.
# The four files together contain every cleaned row exactly once.
import numpy as np

shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
data_parts = np.array_split(shuffled_df, 4)

for i, part in enumerate(data_parts, start=1):
    divided_file = data_folder / f"divided_data_{i}.csv"
    part.to_csv(divided_file, index=False)
    print(f"Divided data saved to: {divided_file} ({len(part)} rows)")
# 7. Divide the data into 2 equal parts: Training and Validation
num_parts = 2
chunk_size = len(shuffled_df) // num_parts

# Part 1: Training Data
train_part = shuffled_df.iloc[0:chunk_size]
train_output_file = data_folder / "Training data.csv"
train_part.to_csv(train_output_file, index=False)
print(f"Divided data saved to: {train_output_file} ({len(train_part)} rows)")

# Part 2: Validation Data
val_part = shuffled_df.iloc[chunk_size:]
val_output_file = data_folder / "Validation data.csv"
val_part.to_csv(val_output_file, index=False)
print(f"Divided data saved to: {val_output_file} ({len(val_part)} rows)")
