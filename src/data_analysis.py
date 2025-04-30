""":doc
Analyzes the two provided CSV datasets for the language family prediction task.

Loads the data, displays headers, basic info, and relationships.
"""

import pandas as pd

# Define file paths relative to the workspace root
DATA1_PATH = "part1/data/CS_assignment3_data_1.csv"
DATA2_PATH = "part1/data/CS_assignment3_data_2.csv"

def analyze_data():
    """Loads and analyzes the two datasets."""
    print(f"Loading data from {DATA1_PATH}...")
    try:
        df1 = pd.read_csv(DATA1_PATH)
        print("Successfully loaded data 1.")
    except FileNotFoundError:
        print(f"Error: File not found at {DATA1_PATH}")
        return
    except Exception as e:
        print(f"Error loading {DATA1_PATH}: {e}")
        return

    print(f"\nLoading data from {DATA2_PATH}...")
    try:
        df2 = pd.read_csv(DATA2_PATH)
        print("Successfully loaded data 2.")
    except FileNotFoundError:
        print(f"Error: File not found at {DATA2_PATH}")
        return
    except Exception as e:
        print(f"Error loading {DATA2_PATH}: {e}")
        return

    print("\n--- Data File 1 Analysis ---")
    print("Header (Columns):")
    print(df1.columns.tolist())
    print("\nFirst 5 rows:")
    print(df1.head())
    print("\nInfo:")
    df1.info(verbose=True, show_counts=True)
    # print("\nDescriptive Statistics (Object columns might be less informative here):")
    # print(df1.describe(include='all')) # Describe might be too verbose for features
    print(f"\nNumber of unique languages: {df1['LanguageName'].nunique()}")
    print(f"Total number of sounds (rows): {len(df1)}")

    print("\n\n--- Data File 2 Analysis ---")
    print("Header (Columns):")
    print(df2.columns.tolist())
    print("\nFirst 5 rows:")
    print(df2.head())
    print("\nInfo:")
    df2.info(verbose=True, show_counts=True)
    # print("\nDescriptive Statistics:") # Only 2 columns, value counts are better
    # print(df2.describe(include='all'))
    print(f"\nNumber of unique languages: {df2['language'].nunique()}")
    print("\nLanguage Family Distribution:")
    print(df2['family'].value_counts())

    print("\n\n--- Relationship Analysis ---")
    # Check if the languages in df2 are a subset of languages in df1
    languages1 = set(df1['LanguageName'].unique())
    languages2 = set(df2['language'].unique())

    print(f"Number of unique languages in Data 1 (using 'LanguageName'): {len(languages1)}")
    print(f"Number of unique languages in Data 2 (using 'language'): {len(languages2)}")
    print(f"Are all languages in Data 2 present in Data 1? {languages2.issubset(languages1)}")
    print(f"Number of languages common to both datasets: {len(languages1.intersection(languages2))}")
    print(f"Languages in Data 2 but not in Data 1: {languages2 - languages1}")
    print(f"Languages in Data 1 but not in Data 2: {languages1 - languages2}") # Might be large

    # The likely relationship is that df1 contains sounds for languages,
    # and df2 provides the family for each language.
    # We'll need to merge df1 and df2 on 'LanguageName' and 'language' respectively.
    print("\nConclusion: Data 1 contains phonological features for sounds per language (using 'LanguageName').")
    print("Data 2 contains the language family for each language (using 'language').")
    print("To train a model, we will merge these two datasets, joining df1's 'LanguageName' with df2's 'language'.")

if __name__ == "__main__":
    analyze_data() 