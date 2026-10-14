import pandas as pd
from pathlib import Path

# Define file paths relative to the repository root
data_folder = Path("data files")
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
