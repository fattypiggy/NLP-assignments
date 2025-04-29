import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_feature_data():
    """Load feature data and feature names"""
    # Load feature data
    X_train = np.load('../processed_data/X_feat_train.npy')
    X_test = np.load('../processed_data/X_feat_test.npy')
    
    # Load feature names
    with open('../processed_data/feature_cols.txt', 'r') as f:
        feature_names = f.read().splitlines()
    
    return X_train, X_test, feature_names

def display_basic_info(X_train, X_test, feature_names):
    """Display basic information about the feature data"""
    print("\n=== Basic Information ===")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features: {len(feature_names)}")
    print("\nFeature names:")
    for i, name in enumerate(feature_names):
        print(f"{i+1}. {name}")

def display_feature_values(X_train, X_test, feature_names):
    """Display the count of 0s and 1s for each feature"""
    print("\n=== Feature Value Counts ===")
    
    # Create DataFrame for easier analysis
    train_df = pd.DataFrame(X_train, columns=feature_names)
    test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Calculate value counts for each feature
    for feature in feature_names:
        print(f"\nFeature: {feature}")
        
        # Training set
        train_counts = train_df[feature].value_counts().sort_index()
        print("Training set:")
        for value, count in train_counts.items():
            print(f"  {value}: {count}")
        
        # Test set
        test_counts = test_df[feature].value_counts().sort_index()
        print("Test set:")
        for value, count in test_counts.items():
            print(f"  {value}: {count}")
        
        # Calculate percentages
        train_total = len(train_df)
        test_total = len(test_df)
        print("\nPercentages:")
        print("Training set:")
        for value, count in train_counts.items():
            print(f"  {value}: {count/train_total*100:.2f}%")
        print("Test set:")
        for value, count in test_counts.items():
            print(f"  {value}: {count/test_total*100:.2f}%")

def plot_feature_distributions(X_train, X_test, feature_names):
    """Plot distributions of features"""
    print("\n=== Plotting Feature Distributions ===")
    
    # Create figure with subplots
    n_features = len(feature_names)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 4*n_rows))
    
    for i, feature in enumerate(feature_names):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Plot histograms
        plt.hist(X_train[:, i], alpha=0.5, label='Train', bins=20)
        plt.hist(X_test[:, i], alpha=0.5, label='Test', bins=20)
        
        plt.title(feature)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('../results/feature_distributions.png')
    plt.close()

def plot_feature_correlations(X_train, feature_names):
    """Plot feature correlation matrix"""
    print("\n=== Plotting Feature Correlations ===")
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X_train.T)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(corr_matrix, 
                xticklabels=feature_names,
                yticklabels=feature_names,
                cmap='coolwarm',
                center=0,
                annot=True,
                fmt='.2f',
                square=True)
    
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('../results/feature_correlations.png')
    plt.close()

def display_sample_data(X_train, X_test, feature_names, n_samples=5):
    """Display sample data from both sets"""
    print("\n=== Sample Data ===")
    
    # Create DataFrames
    train_df = pd.DataFrame(X_train[:n_samples], columns=feature_names)
    test_df = pd.DataFrame(X_test[:n_samples], columns=feature_names)
    
    print("\nTraining set samples:")
    print(train_df)
    print("\nTest set samples:")
    print(test_df)

def main():
    # Load data
    X_train, X_test, feature_names = load_feature_data()
    
    # Display basic information
    display_basic_info(X_train, X_test, feature_names)
    
    # Display feature value counts
    display_feature_values(X_train, X_test, feature_names)
    
    # Plot feature distributions
    plot_feature_distributions(X_train, X_test, feature_names)
    
    # Plot feature correlations
    plot_feature_correlations(X_train, feature_names)
    
    # Display sample data
    display_sample_data(X_train, X_test, feature_names)
    
    print("\nAnalysis complete! Check the results directory for plots.")

if __name__ == "__main__":
    main() 