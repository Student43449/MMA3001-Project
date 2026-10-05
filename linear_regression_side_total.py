import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import os
import sys

def run_side_total_regression_with_feature_selection(df_train_path, df_test_path, feature_column_indices_start, feature_column_indices_end, k_features=5):
    """
    Performs Linear Regression to predict 'Side Total' with feature selection.

    Args:
        df_train_path (str): Path to the training data CSV file.
        df_test_path (str): Path to the testing data CSV file.
        feature_column_indices_start (int): The 0-indexed start column for features.
        feature_column_indices_end (int): The 0-indexed end column (exclusive) for features.
        k_features (int): Number of top features to select for 'Side Total'.

    Returns:
        tuple: evaluation metrics, actual/predicted values for both training and validation sets, and selected features.
    """
    try:
        # Load the training and testing data
        df_train_fs = pd.read_csv(df_train_path)
        df_test_fs = pd.read_csv(df_test_path)

        # Define all potential features (X) and target (y)
        feature_cols_all = df_train_fs.columns[feature_column_indices_start : feature_column_indices_end]
        target_col = 'Side Total'

        if target_col not in df_train_fs.columns or target_col not in df_test_fs.columns:
            print("check data set")
            sys.exit(1)

        # Prepare training data
        X_train_fs = df_train_fs[feature_cols_all].apply(pd.to_numeric, errors='coerce')
        y_train_fs = df_train_fs[target_col].apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN values introduced during numeric conversion and align
        aligned_train_fs = pd.concat([X_train_fs, y_train_fs], axis=1).dropna()
        X_train_fs = aligned_train_fs[feature_cols_all]
        y_train_fs = aligned_train_fs[target_col]

        # Prepare testing data
        X_test_fs_all = df_test_fs[feature_cols_all].apply(pd.to_numeric, errors='coerce')
        y_test_fs_target = df_test_fs[target_col].apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN values introduced during numeric conversion and align
        aligned_test_fs = pd.concat([X_test_fs_all, y_test_fs_target], axis=1).dropna()
        X_test_fs_all = aligned_test_fs[feature_cols_all]
        y_test_side_total = aligned_test_fs[target_col]

        if len(X_train_fs) == 0 or len(X_test_fs_all) == 0:
            print("check data set")
            sys.exit(1)

    except Exception as e:
        print("check data set")
        sys.exit(1)

    # Feature Selection for 'Side Total'
    selector_side_total = SelectKBest(f_regression, k=k_features)
    selector_side_total.fit(X_train_fs, y_train_fs)
    selected_features_side_total_list = X_train_fs.columns[selector_side_total.get_support()].tolist()

    # Filter X_train and X_test to include only the selected features for 'Side Total'
    X_train_side_total_selected = X_train_fs[selected_features_side_total_list]
    X_test_side_total_selected = X_test_fs_all[selected_features_side_total_list]

    # Train Linear Regression model for 'Side Total'
    model_side_total = LinearRegression()
    model_side_total.fit(X_train_side_total_selected, y_train_fs)

    # Make predictions on Training Data
    y_pred_train = model_side_total.predict(X_train_side_total_selected)
    mae_train = mean_absolute_error(y_train_fs, y_pred_train)
    r2_train = r2_score(y_train_fs, y_pred_train)

    # Make predictions on Validation/Testing Data
    y_pred_validation = model_side_total.predict(X_test_side_total_selected)
    mae_validation = mean_absolute_error(y_test_side_total, y_pred_validation)
    r2_validation = r2_score(y_test_side_total, y_pred_validation)

    return (
        mae_train, r2_train, y_train_fs, y_pred_train,
        mae_validation, r2_validation, y_test_side_total, y_pred_validation,
        selected_features_side_total_list
    )

if __name__ == "__main__":
    print("--- Side Total Linear Regression Configuration ---")
    
    # Loop until correct training dataset path is provided
    while True:
        train_path = input("Enter path to training dataset (e.g., /content/MMA3001-Project/divided_data_1.csv): ").strip()
        if os.path.exists(train_path):
            break
        print("Error: Training dataset file does not exist. Please check the path and try again.\n")

    # Loop until correct validation dataset path is provided
    while True:
        test_path = input("Enter path to validation/testing dataset (e.g., /content/MMA3001-Project/divided_data_2.csv): ").strip()
        if os.path.exists(test_path):
            break
        print("Error: Validation dataset file does not exist. Please check the path and try again.\n")

    # Run regression assuming features are between indices 1 and 10
    (mae_tr, r2_tr, y_tr_true, y_tr_pred,
     mae_val, r2_val, y_val_true, y_val_pred,
     features) = run_side_total_regression_with_feature_selection(train_path, test_path, 1, 10, k_features=5)

    print("\nSelected Features:", features)

    # Create and display Training error table
    error_tr_df = pd.DataFrame({
        'Metric': ['Mean Absolute Error (MAE)', 'R² Score'],
        'Value': [mae_tr, r2_tr]
    })
    print("\n=== Training Set: Model Evaluation Error Table ===")
    print(error_tr_df.to_string(index=False))

    # Create and display Validation error table
    error_val_df = pd.DataFrame({
        'Metric': ['Mean Absolute Error (MAE)', 'R² Score'],
        'Value': [mae_val, r2_val]
    })
    print("\n=== Validation Set: Model Evaluation Error Table ===")
    print(error_val_df.to_string(index=False))

    # Plot 1: Training Scatterplot with Legend
    plt.figure(figsize=(8, 6))
    plt.scatter(y_tr_true, y_tr_pred, color='green', alpha=0.6, edgecolors='k', label=f'Train: {os.path.basename(train_path)}')
    plt.plot([y_tr_true.min(), y_tr_true.max()], [y_tr_true.min(), y_tr_true.max()], 'r--', lw=2, label='Perfect Fit')
    plt.title('Side Total: Actual Vs Predicited Training')
    plt.xlabel('Actual Side Total')
    plt.ylabel('Predicted Side Total')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot 2: Validation Scatterplot with Legend
    plt.figure(figsize=(8, 6))
    plt.scatter(y_val_true, y_val_pred, color='blue', alpha=0.6, edgecolors='k', label=f'Val: {os.path.basename(test_path)}')
    plt.plot([y_val_true.min(), y_val_true.max()], [y_val_true.min(), y_val_true.max()], 'r--', lw=2, label='Perfect Fit')
    plt.title('Side Total: Actual Vs Predicited Validation')
    plt.xlabel('Actual Side Total')
    plt.ylabel('Predicted Side Total')
    plt.grid(True)
    plt.legend()
    plt.show()
