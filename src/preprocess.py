"""
Preprocesses the sound feature and language family data.

1. Loads the two datasets.
2. Filters out sounds with missing Glottocodes in the features file.
3. Merges the sound features with language family information using Glottocode.
4. Converts phonological features (+/- or 1/0) to binary floats (1.0/0.0).
5. Saves the processed data.
"""

import pandas as pd
import numpy as np
import sys # Import sys to handle potential DtypeWarning explanation

# Define file paths relative to the workspace root
DATA1_PATH = "part1/data/CS_assignment3_data_1.csv"
DATA2_PATH = "part1/data/CS_assignment3_data_2.csv"
OUTPUT_PATH = "part1/data/processed_data.csv" # Output file path

def preprocess_data():
    """Loads, merges, cleans, and transforms the data."""

    print(f"Loading features data from {DATA1_PATH}...")
    try:
        # Add low_memory=False to potentially address DtypeWarning, though it uses more memory
        df1 = pd.read_csv(DATA1_PATH, low_memory=False)
        print("Successfully loaded features data.")
    except FileNotFoundError:
        print(f"Error: File not found at {DATA1_PATH}")
        return None
    except Exception as e:
        print(f"Error loading {DATA1_PATH}: {e}")
        return None

    print(f"Loading family data from {DATA2_PATH}...")
    try:
        df2 = pd.read_csv(DATA2_PATH)
        print("Successfully loaded family data.")
    except FileNotFoundError:
        print(f"Error: File not found at {DATA2_PATH}")
        return None
    except Exception as e:
        print(f"Error loading {DATA2_PATH}: {e}")
        return None

    # --- Data Cleaning and Selection ---
    print("Filtering features data...")
    # Keep only rows where Glottocode is not null
    initial_rows_df1 = len(df1)
    df1_filtered = df1.dropna(subset=['Glottocode']).copy()
    print(f"Removed {initial_rows_df1 - len(df1_filtered)} rows with missing Glottocode from features data.")

    # Select relevant columns from df2
    df2_selected = df2[['Glottocode', 'Family_Glottocode']].copy()
    # Optional: Check for and handle potential duplicates in Glottocode in df2 if needed
    # df2_selected = df2_selected.drop_duplicates(subset=['Glottocode'])
    # Optional: Handle missing Family_Glottocode if necessary (e.g., dropna, fillna)
    # initial_rows_df2 = len(df2_selected)
    # df2_selected = df2_selected.dropna(subset=['Family_Glottocode'])
    # print(f"Removed {initial_rows_df2 - len(df2_selected)} rows with missing Family_Glottocode from family data.")


    # --- Merging Data ---
    print("Merging dataframes on 'Glottocode'...")
    merged_df = pd.merge(df1_filtered, df2_selected, on='Glottocode', how='inner')
    print(f"Merged dataframe shape: {merged_df.shape}")
    if merged_df.empty:
        print("Error: Merged dataframe is empty. Check Glottocodes match between files.")
        return None

    # Check if Family_Glottocode column exists after merge
    if 'Family_Glottocode' not in merged_df.columns:
        print("Error: 'Family_Glottocode' column not found after merge.")
        return None

    # Check for nulls in the target column after merge
    null_targets = merged_df['Family_Glottocode'].isnull().sum()
    if null_targets > 0:
        print(f"Warning: Found {null_targets} rows with missing 'Family_Glottocode' after merge. Consider dropping or imputing.")
        # merged_df.dropna(subset=['Family_Glottocode'], inplace=True)
        # print(f"Dropped rows with missing target. New shape: {merged_df.shape}")


    # --- Feature Conversion ---
    print("Converting features to binary floats...")
    # Identify feature columns (from 'tone' to 'click')
    feature_cols = df1.columns[df1.columns.get_loc('tone'):df1.columns.get_loc('click')+1].tolist()

    processed_count = 0
    error_count = 0
    conversion_map = {'+': 1.0, '1': 1.0, 1: 1.0, '-': 0.0, '0': 0.0, 0: 0.0}

    for col in feature_cols:
        if col in merged_df.columns:
            # Convert column to string first to handle mixed types consistently
            merged_df[col] = merged_df[col].astype(str)
            # Apply the mapping
            original_values = merged_df[col].copy()
            merged_df[col] = merged_df[col].map(conversion_map)
            # Check for values that were not mapped (became NaN)
            unmapped = original_values[merged_df[col].isna()].unique()
            if len(unmapped) > 0:
                print(f"  Warning: Unmapped values found in column '{col}': {unmapped}. They will be treated as NaN.")
                error_count += merged_df[col].isna().sum() # Count NaNs introduced by failed mapping

            # Optional: Handle NaNs after conversion if needed (e.g., fill with 0.0 or drop rows)
            # merged_df[col] = merged_df[col].fillna(0.0)

            processed_count += 1
        else:
            print(f"  Warning: Feature column '{col}' not found in merged dataframe.")

    print(f"Processed {processed_count} feature columns.")
    if error_count > 0:
         print(f"Found {error_count} entries across all feature columns that couldn't be mapped to 1.0 or 0.0 and are now NaN.")

    # Select final columns (features + target)
    final_columns = feature_cols + ['Family_Glottocode']
    # Ensure all selected columns exist
    final_columns = [col for col in final_columns if col in merged_df.columns]
    if 'Family_Glottocode' not in final_columns:
        print("Error: Target column 'Family_Glottocode' is missing.")
        return None

    processed_df = merged_df[final_columns].copy()

    # Optional: Drop rows with any NaN values in features or target
    initial_rows_final = len(processed_df)
    processed_df.dropna(inplace=True)
    print(f"Dropped {initial_rows_final - len(processed_df)} rows containing NaN values after feature conversion or in target.")


    # --- Save Processed Data ---
    print(f"Saving processed data to {OUTPUT_PATH}...")
    try:
        processed_df.to_csv(OUTPUT_PATH, index=False)
        print("Successfully saved processed data.")
    except Exception as e:
        print(f"Error saving processed data: {e}")
        return None

    print("Preprocessing Complete.")
    print(f"Final processed data shape: {processed_df.shape}")
    print(f"Columns: {processed_df.columns.tolist()}")
    print("First 5 rows of processed data:")
    print(processed_df.head())

    return processed_df # Return the dataframe for potential further use

if __name__ == "__main__":
    # Explain DtypeWarning if it occurs
    print("Note: You might see a 'DtypeWarning'. This usually means some columns have mixed data types (e.g., numbers and text).")
    print("      We attempt to handle this during feature conversion, but be aware of potential data inconsistencies.")
    print("      Using 'low_memory=False' during loading might help but increases memory usage.")
    print("-" * 30)

    processed_data = preprocess_data()

    if processed_data is not None:
        # You can add further steps here if needed
        pass
    else:
        print("Preprocessing failed.")
        sys.exit(1) # Exit with error code if preprocessing failed 