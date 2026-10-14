import pandas as pd
import numpy as np
import os

def process_data(raw_path, output_path):
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
    
    # Save result
    cleaned_attempt.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path} with shape {cleaned_attempt.shape}")

if __name__ == '__main__':
    # Default paths within the repo structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, 'data files', 'data.csv')
    output_path = os.path.join(base_dir, 'data files', 'cleaned_data_recreated.csv')
    process_data(raw_path, output_path)
