"""
Prepares the processed data for model training.

1. Loads the processed data.
2. Encodes the target variable ('Family_Glottocode') into numerical labels.
3. Performs a stratified train-test split (80/20).
4. Applies resampling (e.g., RandomOverSampler) ONLY to the training set
   to address class imbalance.
5. Optionally saves or returns the prepared datasets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Ensure imbalanced-learn is installed: pip install imbalanced-learn
try:
    # Using RandomOverSampler as a simple example.
    # Consider others like SMOTE, ADASYN, or RandomUnderSampler, or combinations.
    from imblearn.over_sampling import RandomOverSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    print("Warning: imbalanced-learn library not found. Resampling step will be skipped.")
    print("Install it using: pip install imbalanced-learn")
    IMBLEARN_AVAILABLE = False
import numpy as np
import sys
import joblib # Import joblib for saving the encoder
import os # Import os for path joining

# Define file path relative to the workspace root
PROCESSED_DATA_PATH = "part1/data/processed_data.csv"
TARGET_COLUMN = 'Family_Glottocode'

# Define output paths for prepared data
OUTPUT_DIR = "part1/data/prepared"
PREPARED_DATA_FILE = os.path.join(OUTPUT_DIR, "prepared_data.npz")
LABEL_ENCODER_FILE = os.path.join(OUTPUT_DIR, "label_encoder.joblib")

def prepare_data(file_path, target_col, test_size=0.2, random_state=42, resample=True):
    """
    Loads, encodes, splits, and optionally resamples the data.

    Args:
        file_path (str): Path to the processed data CSV file.
        target_col (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        resample (bool): Whether to apply resampling to the training data.

    Returns:
        tuple: (X_train, y_train, X_test, y_test, label_encoder)
               Returns the processed training and testing sets and the fitted LabelEncoder.
               X_train and y_train will be resampled if resample=True and imblearn is available.
               Returns None if loading fails.
    """
    print(f"Loading processed data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in the dataframe.")
        return None

    # Separate features (X) and target (y)
    X = df.drop(target_col, axis=1)
    y_raw = df[target_col]
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y_raw.shape}")

    # --- Encode Target Variable ---
    print(f"\nEncoding target variable '{target_col}'...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)
    print(f"Encoded target labels range from 0 to {n_classes - 1}.")
    print(f"Number of unique classes: {n_classes}")
    # Store class names corresponding to encoded labels
    class_names = label_encoder.classes_
    print(f"Example mapping: 0 -> {class_names[0]}, 1 -> {class_names[1]}, ...")


    # --- Stratified Train/Test Split ---
    print(f"\nPerforming stratified train/test split ({1-test_size:.0%}/{test_size:.0%})...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Crucial for maintaining class proportions
        )
        print(f"Original X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"Original X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        print("\nOriginal Training Set Class Distribution (Top 10 & Bottom 5):")
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        train_dist = dict(sorted(zip(label_encoder.inverse_transform(unique_train), counts_train), key=lambda item: item[1], reverse=True))
        print({k: train_dist[k] for i, k in enumerate(train_dist) if i < 10 or i >= len(train_dist) - 5})

        print("\nTest Set Class Distribution (Top 10 & Bottom 5):")
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        test_dist = dict(sorted(zip(label_encoder.inverse_transform(unique_test), counts_test), key=lambda item: item[1], reverse=True))
        print({k: test_dist[k] for i, k in enumerate(test_dist) if i < 10 or i >= len(test_dist) - 5})


    except ValueError as e:
        print(f"\nError during train_test_split: {e}")
        print("This might happen if some classes have too few samples for stratification.")
        print("Consider reducing test_size or merging rare classes.")
        return None


    # --- Resampling (on Training Data Only) ---
    if resample and IMBLEARN_AVAILABLE:
        print("\nApplying resampling to the training data (using RandomOverSampler)...")
        # Define the sampler - alternatives include SMOTE, RandomUnderSampler, etc.
        # Adjust sampling_strategy as needed (e.g., 'minority', 'not majority', or a dict)
        oversampler = RandomOverSampler(sampling_strategy='auto', random_state=random_state)

        try:
            X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
            print(f"Resampled X_train shape: {X_train_resampled.shape}, y_train shape: {y_train_resampled.shape}")

            print("\nResampled Training Set Class Distribution (Sample):")
            unique_resampled, counts_resampled = np.unique(y_train_resampled, return_counts=True)
            resampled_dist = dict(zip(label_encoder.inverse_transform(unique_resampled), counts_resampled))
            # Print first few and last few to show balance
            print({k: resampled_dist[k] for i, k in enumerate(resampled_dist) if i < 5 or i >= len(resampled_dist) - 5})


            X_train, y_train = X_train_resampled, y_train_resampled # Update train variables

        except Exception as e:
            print(f"\nError during resampling: {e}")
            print("Proceeding with original training data.")

    elif resample and not IMBLEARN_AVAILABLE:
        print("\nSkipping resampling because imbalanced-learn is not installed.")

    else:
        print("\nResampling not requested.")


    print("\nData preparation complete.")
    return X_train, y_train, X_test, y_test, label_encoder


if __name__ == "__main__":
    print("--- Starting Data Preparation ---")
    prepared_data = prepare_data(
        PROCESSED_DATA_PATH,
        TARGET_COLUMN,
        test_size=0.2,
        random_state=42,
        resample=True # Set to True to enable resampling
    )

    if prepared_data is not None:
        X_train, y_train, X_test, y_test, le = prepared_data
        print("\n--- Output Shapes ---")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"Number of classes: {len(le.classes_)}")

        # --- Save Prepared Data --- 
        print(f"\n--- Saving Prepared Data to {OUTPUT_DIR} ---")
        try:
            # Ensure data are NumPy arrays before saving
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values
            if isinstance(X_test, pd.DataFrame):
                X_test = X_test.values
            # y_train and y_test should already be numpy arrays from LabelEncoder and resampling

            np.savez_compressed(PREPARED_DATA_FILE,
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test)
            print(f"Saved data arrays to {PREPARED_DATA_FILE}")

            joblib.dump(le, LABEL_ENCODER_FILE)
            print(f"Saved label encoder to {LABEL_ENCODER_FILE}")

            print("\nSaving complete.")

        except Exception as e:
            print(f"\nError saving prepared data: {e}")
            sys.exit(1)

    else:
        print("\nData preparation failed.")
        sys.exit(1)
