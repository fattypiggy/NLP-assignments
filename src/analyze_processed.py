"""
Analyzes the preprocessed data file.

Loads the processed data and displays basic statistics and information,
including target distribution and feature means.
"""

import pandas as pd
import numpy as np

# Define file path relative to the workspace root
PROCESSED_DATA_PATH = "part1/data/processed_data.csv"

def analyze_processed_data(file_path):
    """Loads and analyzes the processed dataset."""
    print(f"Loading processed data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print("Successfully loaded processed data.")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Please ensure you have run the preprocessing script first.")
        return
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    print("\n--- Processed Data Analysis ---")

    print(f"Shape (rows, columns): {df.shape}")

    print("\nInfo (Columns, Data Types, Non-null counts):")
    df.info()

    print("\nFirst 5 rows:")
    print(df.head())

    # Target Variable Analysis
    target_col = 'Family_Glottocode'
    if target_col in df.columns:
        print(f"\nTarget Variable ('{target_col}') Distribution:")
        print(df[target_col].value_counts())
        print(f"\nNumber of unique target classes: {df[target_col].nunique()}")
    else:
        print(f"\nWarning: Target column '{target_col}' not found.")

    # Feature Analysis (Means for binary features)
    print("\nFeature Column Means (Proportion of 1.0s):")
    # Assuming all columns except the target are features
    feature_cols = [col for col in df.columns if col != target_col]
    if feature_cols:
        # Ensure features are numeric before calculating mean
        numeric_feature_means = df[feature_cols].select_dtypes(include=np.number).mean()
        if not numeric_feature_means.empty:
             print(numeric_feature_means)
        else:
            print("No numeric feature columns found to calculate means.")

        non_numeric_features = df[feature_cols].select_dtypes(exclude=np.number).columns.tolist()
        if non_numeric_features:
            print(f"\nNon-numeric feature columns present: {non_numeric_features}")
    else:
         print("No feature columns found.")


if __name__ == "__main__":
    analyze_processed_data(PROCESSED_DATA_PATH) 