""":doc
Analyzes the two provided CSV datasets for the language family prediction task.

Loads the data, displays headers, basic info, and relationships.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define file paths relative to the workspace root
DATA1_PATH = "part1/data/CS_assignment3_data_1_new.csv"
DATA2_PATH = "part1/data/CS_assignment3_data_2.csv"
OUTPUT_PATH = "part1/data/processed_data_with_family_new.csv"
OUTPUT_NUMERIC_PATH = "part1/data/processed_data_numeric.csv"

def analyze_data():
    """Loads and analyzes the two datasets."""
    print(f"Loading data from {DATA1_PATH}...")
    try:
        df1 = pd.read_csv(DATA1_PATH, low_memory=False)
        print("Successfully loaded data 1.")
    except FileNotFoundError:
        print(f"Error: File not found at {DATA1_PATH}")
        return
    except Exception as e:
        print(f"Error loading {DATA1_PATH}: {e}")
        return

    print(f"\nLoading data from {DATA2_PATH}...")
    try:
        # Try different encodings since UTF-8 failed
        df2 = pd.read_csv(DATA2_PATH, encoding='latin1')
        print("Successfully loaded data 2.")
    except FileNotFoundError:
        print(f"Error: File not found at {DATA2_PATH}")
        return
    except Exception as e:
        print(f"Error loading {DATA2_PATH}: {e}")
        return

    # Clean the datasets by removing rows with empty values in primary keys
    print("\nCleaning datasets...")
    df1_original_len = len(df1)
    df2_original_len = len(df2)
    
    df1 = df1.dropna(subset=['Glottocode'])
    df2 = df2.dropna(subset=['Glottocode', 'Family_Glottocode'])
    
    print(f"Removed {df1_original_len - len(df1)} rows from dataset 1 with missing Glottocode")
    print(f"Removed {df2_original_len - len(df2)} rows from dataset 2 with missing Glottocode or Family_Glottocode")

    # Find common Glottocodes in both datasets
    glottocodes1 = set(df1['Glottocode'].unique())
    glottocodes2 = set(df2['Glottocode'].unique())
    common_glottocodes = glottocodes1.intersection(glottocodes2)
    
    print(f"\nNumber of Glottocodes in Dataset 1: {len(glottocodes1)}")
    print(f"Number of Glottocodes in Dataset 2: {len(glottocodes2)}")
    print(f"Number of common Glottocodes in both datasets: {len(common_glottocodes)}")
    
    # Filter datasets to only include common Glottocodes
    print("\nFiltering datasets to only include common Glottocodes...")
    df1_filtered_len = len(df1)
    df2_filtered_len = len(df2)
    
    df1 = df1[df1['Glottocode'].isin(common_glottocodes)]
    df2 = df2[df2['Glottocode'].isin(common_glottocodes)]
    
    print(f"Removed {df1_filtered_len - len(df1)} rows from dataset 1 (not in common Glottocodes)")
    print(f"Removed {df2_filtered_len - len(df2)} rows from dataset 2 (not in common Glottocodes)")

    # Analyze phonological features (from 'tone' to 'click')
    print("\n--- Phonological Feature Analysis ---")
    feature_columns = df1.columns[11:48]  # from 'tone' to 'click'
    print(f"Analyzing {len(feature_columns)} feature columns from 'tone' to 'click'")
    
    # Create a dictionary to store unique values for each feature
    feature_values = {}
    
    for col in feature_columns:
        # Get unique values and their counts
        value_counts = df1[col].value_counts()
        unique_values = df1[col].unique()
        
        # Store in dictionary
        feature_values[col] = {
            'unique_values': sorted(unique_values, key=str),
            'counts': value_counts
        }
        
        # Print summary
        print(f"\nFeature: {col}")
        print(f"  Unique values: {sorted(unique_values, key=str)}")
        print(f"  Value counts:")
        print(value_counts)
    
    # Summarize types of values across all features
    all_unique_values = set()
    for feature in feature_values:
        all_unique_values.update(feature_values[feature]['unique_values'])
    
    print("\n--- Summary of All Unique Values Across Features ---")
    print(f"Total unique values found: {len(all_unique_values)}")
    print(f"All unique values: {sorted(all_unique_values, key=str)}")
    
    # Count how many features have each type of value
    value_in_features = {val: 0 for val in all_unique_values}
    for feature in feature_values:
        for val in feature_values[feature]['unique_values']:
            value_in_features[val] += 1
    
    print("\nNumber of features containing each value type:")
    for val, count in sorted(value_in_features.items(), key=lambda x: (x[1], str(x[0])), reverse=True):
        print(f"  {val}: appears in {count} features")
    
    # Count common value patterns
    print("\n--- Common Value Patterns ---")
    value_pattern_counts = {}
    for feature in feature_values:
        unique_vals = tuple(sorted(feature_values[feature]['unique_values'], key=str))
        if unique_vals in value_pattern_counts:
            value_pattern_counts[unique_vals] += 1
        else:
            value_pattern_counts[unique_vals] = 1
    
    print("Patterns of unique values in features:")
    for pattern, count in sorted(value_pattern_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 1:  # Only show patterns that occur in multiple features
            print(f"  {pattern}: appears in {count} features")

    print("\n--- Data File 1 Analysis (After Filtering) ---")
    print("Header (Columns):")
    print(df1.columns.tolist())
    print("\nFirst 5 rows:")
    print(df1.head())
    print("\nInfo:")
    df1.info(verbose=True, show_counts=True)
    
    # Group by Glottocode to show distribution
    glottocode_counts = df1['Glottocode'].value_counts()
    print(f"\nNumber of unique languages (Glottocode): {len(glottocode_counts)}")
    print(f"Total number of sounds (rows): {len(df1)}")
    print("\nTop 10 languages by number of sounds:")
    print(glottocode_counts.head(10))

    print("\n\n--- Data File 2 Analysis (After Filtering) ---")
    print("Header (Columns):")
    print(df2.columns.tolist())
    print("\nFirst 5 rows:")
    print(df2.head())
    print("\nInfo:")
    df2.info(verbose=True, show_counts=True)
    
    family_counts = df2['Family_Glottocode'].value_counts()
    print(f"\nNumber of unique language families: {len(family_counts)}")
    print("\nTop 15 language families by number of languages:")
    print(family_counts.head(15))

    # Verify that filtering worked correctly
    glottocodes1_after = set(df1['Glottocode'].unique())
    glottocodes2_after = set(df2['Glottocode'].unique())
    
    print("\n--- Verification ---")
    print(f"Number of unique Glottocodes in Dataset 1 after filtering: {len(glottocodes1_after)}")
    print(f"Number of unique Glottocodes in Dataset 2 after filtering: {len(glottocodes2_after)}")
    print(f"Are all Glottocodes in Dataset 1 present in Dataset 2? {glottocodes1_after.issubset(glottocodes2_after)}")
    print(f"Are all Glottocodes in Dataset 2 present in Dataset 1? {glottocodes2_after.issubset(glottocodes1_after)}")
    
    # These should be equal after filtering
    if len(glottocodes1_after) != len(glottocodes2_after):
        print("WARNING: The number of unique Glottocodes in the two datasets is not the same after filtering!")
    else:
        print(f"SUCCESS: Both datasets now contain exactly {len(glottocodes1_after)} unique Glottocodes.")

    # Create a mapping from Glottocode to Family_Glottocode
    print("\n--- Creating Merged Dataset with Family Labels ---")
    # Get a clean mapping of Glottocode to Family_Glottocode
    glottocode_to_family = dict(zip(df2['Glottocode'], df2['Family_Glottocode']))
    
    # Add Family_Glottocode to df1
    df1['Family_Glottocode'] = df1['Glottocode'].map(glottocode_to_family)
    
    # Verify all rows have a family
    missing_family = df1['Family_Glottocode'].isna().sum()
    if missing_family > 0:
        print(f"WARNING: {missing_family} rows ({missing_family/len(df1)*100:.2f}%) are missing family labels!")
    else:
        print("SUCCESS: All rows have family labels.")
    
    # Save original merged dataset
    output_dir = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    df1.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved merged dataset to {OUTPUT_PATH} with {len(df1)} rows and {len(df1.columns)} columns.")
    
    # Display sample of the merged data
    print("\nSample of merged data (first 5 rows):")
    print(df1[['Glottocode', 'LanguageName', 'Family_Glottocode']].head())
    
    # Count distribution of families in the merged dataset
    merged_family_counts = df1['Family_Glottocode'].value_counts()
    print(f"\nNumber of unique language families in merged dataset: {len(merged_family_counts)}")
    print("\nTop 15 language families by number of sounds in merged dataset:")
    print(merged_family_counts.head(15))
    
    # Preprocess feature data: convert '+', '-', '0' to numeric values
    print("\n--- Preprocessing Feature Values ---")
    # Create a new DataFrame to store preprocessed data
    df_numeric = df1.copy()
    
    # Conversion function
    def convert_to_numeric(value):
        if value == '+':
            return 1.0
        elif value == '-' or value == '0':
            return 0.0
        else:
            # For compound values (like '+,-', '-,+', etc.), use the rule of the first character
            if isinstance(value, str) and len(value) > 0:
                first_char = value[0]
                if first_char == '+':
                    return 1.0
                else:  # '-' or other
                    return 0.0
            return np.nan
    
    # Process all feature columns
    for col in feature_columns:
        print(f"Processing feature: {col}")
        df_numeric[col] = df_numeric[col].apply(convert_to_numeric)
    
    # Check for missing values
    null_count = df_numeric[feature_columns].isnull().sum().sum()
    if null_count > 0:
        print(f"Warning: {null_count} missing values in numeric features, filling with 0.0.")
        df_numeric = df_numeric.fillna(0.0)
    
    # Summarize feature distribution
    print("\nFeature distribution summary:")
    feature_stats = pd.DataFrame({
        'Feature Name': feature_columns,
        'Mean': [df_numeric[col].mean() for col in feature_columns],
        'Proportion of 1s': [df_numeric[col].mean() for col in feature_columns],
        'Proportion of 0s': [1 - df_numeric[col].mean() for col in feature_columns]
    })
    print(feature_stats.head(10))
    
    # Save preprocessed data
    df_numeric.to_csv(OUTPUT_NUMERIC_PATH, index=False)
    print(f"Saved numeric dataset to {OUTPUT_NUMERIC_PATH} with {len(df_numeric)} rows and {len(df_numeric.columns)} columns.")
    
    # Display sample of preprocessed data
    print("\nSample of preprocessed data (first 5 rows):")
    sample_cols = list(feature_columns[:5]) + ['Family_Glottocode']
    print(df_numeric[sample_cols].head())

    print("\nConclusion: Data 1 contains phonological features for sounds per language (using Glottocode).")
    print("Data 2 contains the language family for each language (using Family_Glottocode).")
    print("The datasets have been filtered to only include languages present in both datasets.")
    print("A new merged dataset has been created with phonological features and family labels.")
    print("An additional preprocessed dataset with numeric features has been created.")

if __name__ == "__main__":
    analyze_data() 