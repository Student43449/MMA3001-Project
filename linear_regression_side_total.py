
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import os

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
        tuple: (mae_side_total, r2_side_total, y_test_side_total, y_pred_side_total, selected_features_side_total_list)
    """
    # Load the training and testing data
    df_train_fs = pd.read_csv(df_train_path)
    df_test_fs = pd.read_csv(df_test_path)

    # Define all potential features (X) and target (y)
    feature_cols_all = df_train_fs.columns[feature_column_indices_start : feature_column_indices_end]
    target_col = 'Side Total'

    # Prepare training data
    X_train_fs = df_train_fs[feature_cols_all].apply(pd.to_numeric, errors='coerce')
    y_train_fs = df_train_fs[target_col].apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values introduced during numeric conversion and align
    aligned_train_fs = pd.concat([X_train_fs, y_train_fs], axis=1).dropna()
    X_train_fs = aligned_train_fs[feature_cols_all]
    y_train_fs = aligned_train_fs[target_col]

    # Prepare testing data (using all initial feature columns)
    X_test_fs_all = df_test_fs[feature_cols_all].apply(pd.to_numeric, errors='coerce')
    y_test_fs_target = df_test_fs[target_col].apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values introduced during numeric conversion and align
    aligned_test_fs = pd.concat([X_test_fs_all, y_test_fs_target], axis=1).dropna()
    X_test_fs_all = aligned_test_fs[feature_cols_all]
    y_test_side_total = aligned_test_fs[target_col]

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

    # Make predictions
    y_pred_side_total = model_side_total.predict(X_test_side_total_selected)

    # Evaluate the model
    mae_side_total = mean_absolute_error(y_test_side_total, y_pred_side_total)
    r2_side_total = r2_score(y_test_side_total, y_pred_side_total)

    return mae_side_total, r2_side_total, y_test_side_total, y_pred_side_total, selected_features_side_total_list
