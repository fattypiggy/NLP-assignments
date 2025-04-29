import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

def load_and_combine_data(file1_path, file2_path):
    """
    Load and combine the two CSV files for NLP assignment 3
    
    Args:
        file1_path: Path to CS_assignment3_data_1.csv (sounds with features)
        file2_path: Path to CS_assignment3_data_2.csv (languages with families)
        
    Returns:
        Combined DataFrame and preprocessed data for model training
    """
    # Load both dataframes
    print(f"Loading data from {file1_path}")
    df1 = pd.read_csv(file1_path, low_memory=False)
    
    print(f"Loading data from {file2_path}")
    df2 = pd.read_csv(file2_path)
    
    print(f"Data 1 shape: {df1.shape}")
    print(f"Data 2 shape: {df2.shape}")
    
    # Display the column names
    print("\nColumns in data 1:")
    print(df1.columns.tolist())
    print("\nColumns in data 2:")
    print(df2.columns.tolist())
    
    # Merge the two dataframes on the common identifier (Glottocode)
    merged_df = pd.merge(df1, df2, on='Glottocode', how='inner')
    
    print(f"\nMerged data shape: {merged_df.shape}")
    
    # Check if LanguageName exists in the merged dataframe
    if 'LanguageName' in merged_df.columns:
        print(f"Number of unique languages: {merged_df['LanguageName'].nunique()}")
    else:
        print("LanguageName column not found in merged dataframe")
    
    # If the column Family_Name exists in df2, we should have it in our merged dataframe
    if 'Family_Name' in merged_df.columns:
        print(f"Number of unique language families: {merged_df['Family_Name'].nunique()}")
        print("Sample language families:", merged_df['Family_Name'].unique()[:5])
    
    return merged_df

def preprocess_data(merged_df):
    """
    Preprocess the merged data for neural network training
    
    Args:
        merged_df: Combined DataFrame from load_and_combine_data
        
    Returns:
        X_features: Feature vectors for Model 1
        X_phonemes: Phoneme data for Model 2 embeddings
        y: Encoded language family labels
        feature_cols: Names of the feature columns
        le: Label encoder for decoding predictions
    """
    # Identify the feature columns (from 'tone' to 'click')
    feature_start_idx = merged_df.columns.get_loc('tone')
    feature_end_idx = merged_df.columns.get_loc('click')
    feature_cols = merged_df.columns[feature_start_idx:feature_end_idx+1].tolist()
    
    print(f"\nNumber of phonological features: {len(feature_cols)}")
    print("Feature columns:", feature_cols)
    
    # Transform the features: '+' → 1, '-' and '0' → 0
    df_features = merged_df[feature_cols].copy()
    
    # Replace values
    df_features = df_features.replace({'+': 1, '-': 0, '0': 0})
    
    # Convert to numeric type
    for col in feature_cols:
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
    
    # Fill NaN values with 0
    df_features = df_features.fillna(0)
    
    # Extract the features for Model 1
    X_features = df_features.values
    
    # For Model 2, we'll use the phoneme data
    X_phonemes = merged_df['Phoneme'].values
    
    # Encode the language family labels
    le = LabelEncoder()
    y = le.fit_transform(merged_df['Family_Name'])
    
    # Verify and fix label range
    unique_labels = np.unique(y)
    print(f"\nUnique labels before adjustment: {unique_labels}")
    print(f"Label range before adjustment: {y.min()} to {y.max()}")
    
    # Create a mapping to ensure labels are consecutive from 0
    label_map = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_labels))}
    y = np.array([label_map[label] for label in y])
    
    print(f"Unique labels after adjustment: {np.unique(y)}")
    print(f"Label range after adjustment: {y.min()} to {y.max()}")
    print(f"Number of unique language families: {len(np.unique(y))}")
    print(f"Sample language families: {le.classes_[:5]}")
    
    print(f"\nFeature data shape: {X_features.shape}")
    print(f"Number of examples: {len(y)}")
    
    return X_features, X_phonemes, y, feature_cols, le

def create_train_test_split(X_features, X_phonemes, y, merged_df, test_size=0.2, random_state=42):
    """
    Create train/test splits for both model approaches, ensuring each language is split 8:2
    
    Args:
        X_features: Feature vectors for Model 1
        X_phonemes: Phoneme data for Model 2
        y: Encoded language family labels
        merged_df: Original merged dataframe containing language information
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Train/test splits for both models
    """
    # Get unique languages
    unique_languages = merged_df['LanguageName'].unique()
    print(f"\nNumber of unique languages: {len(unique_languages)}")
    
    # Initialize arrays for train and test indices
    train_indices = []
    test_indices = []
    
    # Split each language's data
    for lang in unique_languages:
        # Get indices for this language
        lang_indices = merged_df[merged_df['LanguageName'] == lang].index
        
        # Calculate split size
        n_samples = len(lang_indices)
        n_test = int(n_samples * test_size)
        
        # Randomly select test indices
        np.random.seed(random_state)
        test_idx = np.random.choice(lang_indices, size=n_test, replace=False)
        train_idx = np.setdiff1d(lang_indices, test_idx)
        
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)
    
    # Convert to numpy arrays
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    # Shuffle indices
    np.random.seed(random_state)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    # Split the data
    X_feat_train = X_features[train_indices]
    X_feat_test = X_features[test_indices]
    X_phon_train = X_phonemes[train_indices]
    X_phon_test = X_phonemes[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Print statistics
    print(f"\nTraining set size: {len(y_train)}")
    print(f"Test set size: {len(y_test)}")
    print(f"Training set label range: {y_train.min()} to {y_train.max()}")
    print(f"Test set label range: {y_test.min()} to {y_test.max()}")
    
    # Print language distribution
    train_langs = merged_df.iloc[train_indices]['LanguageName'].nunique()
    test_langs = merged_df.iloc[test_indices]['LanguageName'].nunique()
    print(f"\nNumber of languages in training set: {train_langs}")
    print(f"Number of languages in test set: {test_langs}")
    
    # Print sample counts for a few languages
    print("\nSample counts for a few languages:")
    for lang in unique_languages[:5]:
        train_count = len(merged_df.iloc[train_indices][merged_df.iloc[train_indices]['LanguageName'] == lang])
        test_count = len(merged_df.iloc[test_indices][merged_df.iloc[test_indices]['LanguageName'] == lang])
        print(f"{lang}: {train_count} train, {test_count} test")
    
    return X_feat_train, X_feat_test, X_phon_train, X_phon_test, y_train, y_test

def save_processed_data(X_feat_train, X_feat_test, X_phon_train, X_phon_test, 
                       y_train, y_test, feature_cols, le, output_dir='../processed_data'):
    """
    Save the processed data for later use
    
    Args:
        X_feat_train, X_feat_test: Feature data train/test splits
        X_phon_train, X_phon_test: Phoneme data train/test splits
        y_train, y_test: Target labels train/test splits
        feature_cols: Names of the feature columns
        le: Label encoder for language families
        output_dir: Directory to save the data
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the numpy arrays
    np.save(f"{output_dir}/X_feat_train.npy", X_feat_train)
    np.save(f"{output_dir}/X_feat_test.npy", X_feat_test)
    np.save(f"{output_dir}/X_phon_train.npy", X_phon_train)
    np.save(f"{output_dir}/X_phon_test.npy", X_phon_test)
    np.save(f"{output_dir}/y_train.npy", y_train)
    np.save(f"{output_dir}/y_test.npy", y_test)
    
    # Save the feature column names
    with open(f"{output_dir}/feature_cols.txt", 'w') as f:
        f.write('\n'.join(feature_cols))
    
    # Save the label encoder classes
    np.save(f"{output_dir}/label_classes.npy", le.classes_)
    
    print(f"\nProcessed data saved to {output_dir}")

def analyze_class_imbalance(y, le, merged_df, output_dir='../processed_data'):
    """
    Analyze the class distribution to check for imbalance
    
    Args:
        y: Encoded language family labels
        le: Label encoder for decoding predictions
        merged_df: The merged dataframe containing language information
        output_dir: Directory to save the visualization
    """
    # Count instances per class
    class_counts = Counter(y)
    
    # Get the actual class names
    class_names = le.classes_
    
    # Create a dataframe for better analysis
    class_df = pd.DataFrame({
        'family': class_names,
        'count': [class_counts[i] for i in range(len(class_names))],
    })
    
    # Sort by count
    class_df = class_df.sort_values('count', ascending=False)
    
    # Calculate statistics
    total_samples = len(y)
    min_class_size = class_df['count'].min()
    max_class_size = class_df['count'].max()
    imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float('inf')
    
    print("\n=== Class Imbalance Analysis ===")
    print(f"Total number of classes: {len(class_names)}")
    print(f"Total number of samples: {total_samples}")
    print(f"Minimum class size: {min_class_size} (class: {class_df.iloc[-1]['family']})")
    print(f"Maximum class size: {max_class_size} (class: {class_df.iloc[0]['family']})")
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    # Print top 5 largest and smallest classes
    print("\nTop 5 largest classes:")
    for i in range(min(5, len(class_df))):
        row = class_df.iloc[i]
        print(f"  {row['family']}: {row['count']} samples ({row['count']/total_samples*100:.2f}%)")
    
    print("\nTop 5 smallest classes:")
    for i in range(min(5, len(class_df))):
        row = class_df.iloc[-(i+1)]
        print(f"  {row['family']}: {row['count']} samples ({row['count']/total_samples*100:.2f}%)")
    
    # Determine if there's significant imbalance
    if imbalance_ratio > 10:
        print("\nWARNING: Significant class imbalance detected. Consider using techniques like:")
        print("  - Class weights")
        print("  - Oversampling minority classes")
        print("  - Undersampling majority classes")
    elif imbalance_ratio > 3:
        print("\nNOTE: Moderate class imbalance detected. May need to address during model training.")
    else:
        print("\nClass distribution is relatively balanced.")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Visualize class distribution
        plt.figure(figsize=(12, 8))
        
        # Plot top 20 classes for better visibility
        top_n = min(20, len(class_df))
        subset_df = class_df.head(top_n)
        
        # Convert class names to strings explicitly to avoid TypeError
        family_names = subset_df['family'].astype(str).tolist()
        counts = subset_df['count'].tolist()
        
        # Create horizontal bar chart with string labels
        bars = plt.barh(range(len(family_names)), counts)
        plt.yticks(range(len(family_names)), family_names)
        
        plt.xlabel('Number of samples')
        plt.ylabel('Language Family')
        plt.title(f'Distribution of Top {top_n} Language Families')
        
        # Add count labels to the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.5, i, f'{width}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/class_distribution.png")
        print(f"\nClass distribution visualization saved to {output_dir}/class_distribution.png")
    except Exception as e:
        print(f"\nWarning: Could not create visualization: {str(e)}")
        print("Continuing with data preparation...")
    
    return class_df

def main():
    """
    Main function to demonstrate data preparation
    """
    # Define file paths
    file1_path = "../data/CS_assignment3_data_1.csv"
    file2_path = "../data/CS_assignment3_data_2.csv"
    
    # Load and combine the data
    merged_df = load_and_combine_data(file1_path, file2_path)
    
    # Preprocess the data
    X_features, X_phonemes, y, feature_cols, le = preprocess_data(merged_df)
    
    # Analyze class imbalance
    class_df = analyze_class_imbalance(y, le, merged_df)
    
    # Create train/test splits
    X_feat_train, X_feat_test, X_phon_train, X_phon_test, y_train, y_test = create_train_test_split(
        X_features, X_phonemes, y, merged_df
    )
    
    # Save the processed data
    save_processed_data(
        X_feat_train, X_feat_test, X_phon_train, X_phon_test, 
        y_train, y_test, feature_cols, le
    )
    
    print("\nData preparation complete!")
    
    # Example usage for models
    print("\nExample usage for models:")
    print("1. For Model 1 (feature-based):")
    print("   - Input shape:", X_feat_train.shape)
    print("   - Number of features:", X_feat_train.shape[1])
    print("   - Number of classes:", len(np.unique(y_train)))
    
    print("\n2. For Model 2 (embedding-based):")
    print("   - Need to create embeddings for unique phonemes:", len(np.unique(X_phon_train)))
    print("   - Same number of classes:", len(np.unique(y_train)))

if __name__ == "__main__":
    main() 