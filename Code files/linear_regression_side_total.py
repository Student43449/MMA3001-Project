"""Establish a linear-regression baseline for predicting golf-shot Side Total.

The script selects a configurable number of input features using SelectKBest
with f_regression, fits a linear regression model, reports MAE and R-squared
for training and validation data, and plots actual versus predicted values.
It expects the training and validation CSV files in the repository's data
files directory; run Data_processing.py first.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import os
from pathlib import Path

def run_side_total_regression_with_feature_selection(
    df_train_path,
    df_test_path,
    feature_columns,
    k_features=5,
):
    """Fit and evaluate a feature-selected linear regression model for Side Total.

    Args:
        df_train_path (str or pathlib.Path): Path to the training CSV file.
        df_test_path (str or pathlib.Path): Path to the validation/test CSV file.
        feature_columns (list[str]): Explicit names of candidate predictor
            columns. Target columns must not be included.
        k_features (int): Number of candidate features to retain, ranked using
            univariate linear-regression tests.

    Returns:
        tuple: Training MAE and R-squared, training actual and predicted values,
        validation MAE and R-squared, validation actual and predicted values,
        and the list of selected feature names.

    Raises:
        SystemExit: If the input data cannot be loaded, required target columns
            are missing, or no usable rows remain after numeric conversion.
    """
    # Load training and validation data, preserving informative error messages.
    try:
        df_train_fs = pd.read_csv(df_train_path)
        df_test_fs = pd.read_csv(df_test_path)
    except Exception as exc:
        raise RuntimeError(f"Could not load training or validation data: {exc}") from exc

    target_col = "Side Total"
    if not feature_columns:
        raise ValueError("Provide at least one candidate feature column.")
    if target_col in feature_columns or "Total" in feature_columns:
        raise ValueError("Target columns must not be included among the input features.")

    # Validate both datasets explicitly so column-order changes cannot alter inputs.
    required_columns = list(feature_columns) + [target_col]
    for dataset_name, dataset in (("training", df_train_fs), ("validation", df_test_fs)):
        missing = [column for column in required_columns if column not in dataset.columns]
        if missing:
            raise ValueError(f"The {dataset_name} dataset is missing columns: {missing}")

    # Convert selected predictors and the target to numeric values.
    X_train_fs = df_train_fs.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    y_train_fs = pd.to_numeric(df_train_fs[target_col], errors="coerce")
    X_test_fs_all = df_test_fs.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    y_test_fs_target = pd.to_numeric(df_test_fs[target_col], errors="coerce")

    # Drop rows with missing/non-numeric values while keeping predictors and targets aligned.
    aligned_train_fs = pd.concat([X_train_fs, y_train_fs.rename(target_col)], axis=1).dropna()
    aligned_test_fs = pd.concat([X_test_fs_all, y_test_fs_target.rename(target_col)], axis=1).dropna()

    X_train_fs = aligned_train_fs.loc[:, feature_columns]
    y_train_fs = aligned_train_fs[target_col]
    X_test_fs_all = aligned_test_fs.loc[:, feature_columns]
    y_test_side_total = aligned_test_fs[target_col]

    if aligned_train_fs.empty or aligned_test_fs.empty:
        raise ValueError("No complete numeric rows remain in the training or validation dataset.")

    if not 1 <= k_features <= len(feature_columns):
        raise ValueError(
            f"k_features must be between 1 and {len(feature_columns)}; got {k_features}."
        )

    # Feature Selection for 'Side Total'
    # Rank candidate features using their individual linear relationship with Side Total.
    # Fit the selector on training data only to avoid using validation outcomes for selection.
    selector_side_total = SelectKBest(f_regression, k=k_features)
    selector_side_total.fit(X_train_fs, y_train_fs)
    selected_features_side_total_list = X_train_fs.columns[selector_side_total.get_support()].tolist()

    # Filter X_train and X_test to include only the selected features for 'Side Total'
    X_train_side_total_selected = X_train_fs[selected_features_side_total_list]
    X_test_side_total_selected = X_test_fs_all[selected_features_side_total_list]

    # Train Linear Regression model for 'Side Total'
    model_side_total = LinearRegression()
    # Fit the regression coefficients using only the selected training features.
    model_side_total.fit(X_train_side_total_selected, y_train_fs)

    # Make predictions on Training Data
    y_pred_train = model_side_total.predict(X_train_side_total_selected)
    # MAE measures average absolute prediction error; R² measures explained variation.
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
    # Resolve data paths relative to this script so the current working directory
    # does not need to be the repository root.
    repo_root = Path(__file__).resolve().parent.parent
    data_folder = repo_root / "data files"

    train_path = data_folder / "Training data.csv"
    test_path = data_folder / "Validation data.csv"

    if not train_path.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}. "
            "Run Data_processing.py first."
        )

    if not test_path.is_file():
        raise FileNotFoundError(
            f"Validation dataset not found: {test_path}. "
            "Run Data_processing.py first."
        )

    # Candidate predictors are named explicitly; neither outcome is an input.
    candidate_features = [
        "Club Speed", "Attack Angle", "Club Path", "Low Point", "Swing Plane",
        "Swing Direction", "Dyn. Loft", "Face Angle", "Face To Path", "Ball Speed",
        "Smash Factor", "Launch Angle", "Launch Direction", "Spin Rate", "Spin Axis",
    ]
    (mae_tr, r2_tr, y_tr_true, y_tr_pred,
     mae_val, r2_val, y_val_true, y_val_pred,
     features) = run_side_total_regression_with_feature_selection(
        train_path, test_path, candidate_features, k_features=5
    )

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
    # Points on this reference line have identical actual and predicted values.
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
