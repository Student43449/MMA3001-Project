import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def plot_golf_data_scatter(data_df, output_dir='Raw_data_plots'):
    """Generates four scatter plots with linear regression lines and prints R² and MSE.

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

    plots = [
        ('Club Speed', 'Total', 'Club Speed vs. Total Distance',
         'Club Speed (mph)', 'Total Distance (yds)', 'raw_data_plot_1.jpeg'),
        ('Face Angle', 'Side Total', 'Face Angle vs. Side Total',
         'Face Angle (deg)', 'Side Total (yds)', 'raw_data_plot_2.jpeg'),
        ('Smash Factor', 'Total', 'Smash Factor vs. Total Distance',
         'Smash Factor', 'Total Distance (yds)', 'raw_data_plot_3.jpeg'),
        ('Face To Path', 'Side Total', 'Face To Path vs. Side Total',
         'Face To Path (deg)', 'Side Total (yds)', 'raw_data_plot_4.jpeg')
    ]

    print("\nRegression Results")
    print("------------------")

    for i, (x_col, y_col, title, x_label, y_label, filename) in enumerate(plots, start=1):
        plot_df = df_plot_func[[x_col, y_col]].dropna()

        X = plot_df[[x_col]]
        y = plot_df[y_col]

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)

        print(f"Plot {i} - {title}")
        print(f"R²  = {r2:.4f}")
        print(f"MSE = {mse:.4f}")
        print()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x_col, y=y_col, data=plot_df)

        # Plot the fitted linear regression line.
        x_line = pd.DataFrame({x_col: [X[x_col].min(), X[x_col].max()]})
        y_line = model.predict(x_line)
        plt.plot(x_line[x_col], y_line, linewidth=2, label='Linear regression')

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    print(f"Four scatter plots with regression lines saved as JPEG files in '{output_dir}'.")
