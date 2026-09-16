
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_golf_data_scatter(data_df, output_dir='MMA3001-Project'):
    """Generates four scatter plots from the golf data DataFrame and saves them as JPEG files.

    Args:
        data_df (pd.DataFrame): The DataFrame containing golf swing data.
        output_dir (str): The directory where the plots will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_plot_func = data_df.copy()

    numeric_cols = [
        'Club Speed', 'Total', 'Face Angle', 'Side Total',
        'Smash Factor', 'Face To Path'
    ]
    for col in numeric_cols:
        if col in df_plot_func.columns:
            df_plot_func[col] = pd.to_numeric(df_plot_func[col], errors='coerce')

    df_plot_func = df_plot_func.dropna(subset=numeric_cols)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Club Speed', y='Total', data=df_plot_func)
    plt.title('Club Speed vs. Total Distance')
    plt.xlabel('Club Speed (mph)')
    plt.ylabel('Total Distance (yds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'raw_data_plot_1.jpeg'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Face Angle', y='Side Total', data=df_plot_func)
    plt.title('Face Angle vs. Side Total')
    plt.xlabel('Face Angle (deg)')
    plt.ylabel('Side Total (yds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'raw_data_plot_2.jpeg'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Smash Factor', y='Total', data=df_plot_func)
    plt.title('Smash Factor vs. Total Distance')
    plt.xlabel('Smash Factor')
    plt.ylabel('Total Distance (yds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'raw_data_plot_3.jpeg'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Face To Path', y='Side Total', data=df_plot_func)
    plt.title('Face To Path vs. Side Total')
    plt.xlabel('Face To Path (deg)')
    plt.ylabel('Side Total (yds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'raw_data_plot_4.jpeg'))
    plt.close()

    print(f"Four scatter plots saved as JPEG files in '{output_dir}'.")
