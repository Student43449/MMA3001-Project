
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_golf_data_scatter(data_df):
    """Generates four scatter plots from the golf data DataFrame.

    Args:
        data_df (pd.DataFrame): The DataFrame containing golf swing data.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_plot_func = data_df.copy()

    # Ensure the columns are numeric for plotting
    numeric_cols = [
        'Club Speed', 'Total', 'Face Angle', 'Side Total',
        'Smash Factor', 'Face To Path'
    ]
    for col in numeric_cols:
        if col in df_plot_func.columns:
            df_plot_func[col] = pd.to_numeric(df_plot_func[col], errors='coerce')

    # Drop any rows that might have become NaN after numeric conversion for these specific columns
    df_plot_func = df_plot_func.dropna(subset=numeric_cols)

    # Plot 1: Club Speed vs Total
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Club Speed', y='Total', data=df_plot_func)
    plt.title('Club Speed vs. Total Distance')
    plt.xlabel('Club Speed (mph)')
    plt.ylabel('Total Distance (yds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Plot 2: Face Angle vs Side Total
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Face Angle', y='Side Total', data=df_plot_func)
    plt.title('Face Angle vs. Side Total')
    plt.xlabel('Face Angle (deg)')
    plt.ylabel('Side Total (yds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Plot 3: Smash Factor vs Total
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Smash Factor', y='Total', data=df_plot_func)
    plt.title('Smash Factor vs. Total Distance')
    plt.xlabel('Smash Factor')
    plt.ylabel('Total Distance (yds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # Plot 4: Face To Path vs Side Total
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Face To Path', y='Side Total', data=df_plot_func)
    plt.title('Face To Path vs. Side Total')
    plt.xlabel('Face To Path (deg)')
    plt.ylabel('Side Total (yds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
