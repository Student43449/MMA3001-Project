import pandas as pd
import numpy as np
import os
import glob

def process_data(raw_path, train_output, val_output):
    # Load raw, skipping the first row (the units row)
    raw_df = pd.read_csv(raw_path)
    cleaned_attempt = raw_df.drop(index=0)

    # Columns to keep
    columns_to_keep = [
        'Club', 'Club Speed', 'Attack Angle', 'Club Path', 'Low Point', 
        'Swing Plane', 'Swing Direction', 'Dyn. Loft', 'Face Angle', 
        'Face To Path', 'Ball Speed', 'Smash Factor', 'Launch Angle', 
        'Launch Direction', 'Spin Rate', 'Spin Axis', 'Max Height - Dist.', 
        'Max Height - Height', 'Max Height - Side', 'Last data Point - Length', 
        'Last data Point - Side', 'Last data Point - Height', 'Last data Point - Time', 
        'Carry Flat - Length', 'Carry Flat - Side', 'Carry Flat - Land. Angle', 
        'Carry Flat - Ball Speed', 'Carry Flat - Time', 'Total', 'Side Total', 'Curve'
    ]

    cleaned_attempt = cleaned_attempt[columns_to_keep]

    # Convert all columns except 'Club' to numeric
    for col in columns_to_keep:
        if col != 'Club':
            cleaned_attempt[col] = pd.to_numeric(cleaned_attempt[col], errors='coerce')

    # Drop rows with NaN values
    cleaned_attempt = cleaned_attempt.dropna().reset_index(drop=True)
    
    # Shuffle and split into two equal-sized datasets
    shuffled_df = cleaned_attempt.sample(frac=1, random_state=42).reset_index(drop=True)
    half_size = len(shuffled_df) // 2
    
    train_df = shuffled_df.iloc[:half_size]
    val_df = shuffled_df.iloc[half_size:]
    
    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)
    print(f"Created {train_output} with {len(train_df)} rows.")
    print(f"Created {val_output} with {len(val_df)} rows.")

    # Remove obsolete divided_data_*.csv files
    data_dir = os.path.dirname(raw_path)
    for path in glob.glob(os.path.join(data_dir, 'divided_data_*.csv')):
        try:
            os.remove(path)
            print(f"Removed obsolete file: {path}")
        except Exception as e:
            print(f"Error removing {path}: {e}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, 'data files', 'data.csv')
    train_output = os.path.join(base_dir, 'data files', 'Training_data.csv')
    val_output = os.path.join(base_dir, 'data files', 'Validation_data.csv')
    process_data(raw_path, train_output, val_output)
