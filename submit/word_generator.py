"""
Word Generator using CBOW (Continuous Bag of Words) Model for Character Prediction

This script loads English words from orthodata.csv, builds a character-level CBOW model,
trains it to predict the next character in a word, and generates new words.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import string
import os

# Create directories if they don't exist
os.makedirs("../models", exist_ok=True)

# Parameters
EMBEDDING_DIM = 64
CONTEXT_SIZE = 4  # Number of characters to use as context
HIDDEN_DIM = 128
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

class CBOWCharModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        """
        Initialize the CBOW model for character prediction
        
        Args:
            vocab_size: Size of the character vocabulary
            embedding_dim: Dimension of character embeddings
            context_size: Number of context characters to use
            hidden_dim: Size of hidden layer
        """
        super(CBOWCharModel, self).__init__()
        
        # Character embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Hidden layers
        self.hidden1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer predicts the next character
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, inputs):
        """
        Forward pass of the model
        
        Args:
            inputs: Batch of context characters [batch_size, context_size]
            
        Returns:
            Character prediction logits
        """
        # Get embeddings for all context characters
        embeds = self.embeddings(inputs)  # [batch_size, context_size, embedding_dim]
        
        # Flatten the embeddings of context characters
        embeds = embeds.view(embeds.shape[0], -1)  # [batch_size, context_size * embedding_dim]
        
        # Pass through hidden layers
        hidden1_out = self.relu(self.hidden1(embeds))
        hidden1_out = self.dropout(hidden1_out)
        hidden2_out = self.relu(self.hidden2(hidden1_out))
        hidden2_out = self.dropout(hidden2_out)
        
        # Get output logits
        logits = self.output(hidden2_out)
        
        return logits

class CharDataset(Dataset):
    def __init__(self, words, char_to_idx, context_size):
        """
        Dataset for character-level CBOW model
        
        Args:
            words: List of words to use
            char_to_idx: Dictionary mapping characters to indices
            context_size: Number of characters to use as context
        """
        self.data = []
        self.context_size = context_size
        self.char_to_idx = char_to_idx
        
        # Create training samples from words
        for word in words:
            if len(word) <= context_size:
                continue
                
            # Add start and end tokens to the word
            word = '^' + word + '$'
            
            # Create training samples with sliding window
            for i in range(len(word) - context_size):
                context = word[i:i+context_size]
                target = word[i+context_size]
                
                # Convert characters to indices
                context_ids = [char_to_idx.get(c, char_to_idx['<UNK>']) for c in context]
                target_id = char_to_idx.get(target, char_to_idx['<UNK>'])
                
                self.data.append((context_ids, target_id))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def load_words_from_csv(file_path):
    """
    Load English words from orthodata.csv
    
    Args:
        file_path: Path to the orthodata.csv file
        
    Returns:
        List of English words
    """
    print(f"Attempting to load words from {file_path}")
    words = []
    
    # Try different approaches to read the file
    try:
        # First try: Standard pandas CSV reading
        encodings = ['utf-8', 'latin1', 'cp1252']
        delimiters = [',', '\t', ';']
        
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    print(f"Trying to read with encoding: {encoding}, delimiter: '{delimiter}'")
                    df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter, 
                                     on_bad_lines='skip')  # Updated parameter name
                    
                    # Check for potential word columns (string columns)
                    string_cols = []
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            # Sample a few values to check if they contain alphabetic characters
                            sample = df[col].dropna().head(5)
                            if any(isinstance(val, str) and any(c.isalpha() for c in str(val)) 
                                   for val in sample):
                                string_cols.append(col)
                    
                    if string_cols:
                        # Use the first string column that contains potential words
                        word_col = string_cols[0]
                        print(f"Found potential word column: {word_col}")
                        
                        # Extract words
                        for val in df[word_col].dropna():
                            if isinstance(val, str):
                                # Only keep alphabetic characters
                                word = ''.join(c for c in val if c.isalpha())
                                if word and len(word) >= 2:
                                    words.append(word.lower())
                        
                        if words:
                            print(f"Successfully extracted {len(words)} words")
                            print(f"Sample words: {words[:5]}")
                            return words
                    
                except Exception as e:
                    print(f"Failed with encoding {encoding}, delimiter '{delimiter}': {str(e)}")
    
        # Second try: Read as plain text file
        print("Trying to read as plain text file...")
        encodings = ['utf-8', 'latin1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                    
                    # Process each line
                    for line in lines:
                        # Split by common delimiters
                        for part in line.replace(',', ' ').replace('\t', ' ').replace(';', ' ').split():
                            # Only keep alphabetic characters
                            word = ''.join(c for c in part if c.isalpha())
                            if word and len(word) >= 2:
                                words.append(word.lower())
                    
                    if words:
                        print(f"Successfully extracted {len(words)} words from text file")
                        print(f"Sample words: {words[:5]}")
                        return words
            except Exception as e:
                print(f"Failed to read as text with encoding {encoding}: {str(e)}")
        
        # If we get here, none of the approaches worked
        if not words:
            raise Exception("Could not extract words using any method")
    
    except Exception as e:
        print(f"Error loading words from CSV: {str(e)}")
        
        # If the file really doesn't exist, look for any text files in the data directory
        data_dir = os.path.dirname(file_path)
        if os.path.exists(data_dir):
            print(f"Looking for alternative word files in {data_dir}...")
            for filename in os.listdir(data_dir):
                if filename.endswith('.txt') or filename.endswith('.csv'):
                    alt_path = os.path.join(data_dir, filename)
                    print(f"Found alternative file: {alt_path}")
                    try:
                        with open(alt_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            for line in lines:
                                # Extract words
                                for part in line.split():
                                    word = ''.join(c for c in part if c.isalpha())
                                    if word and len(word) >= 2:
                                        words.append(word.lower())
                        if words:
                            print(f"Successfully extracted {len(words)} words from {filename}")
                            print(f"Sample words: {words[:5]}")
                            return words
                    except Exception as e:
                        print(f"Failed to read alternative file: {str(e)}")
    
    # If everything failed, show error but don't use sample data
    if not words:
        print("ERROR: Could not extract any words from any files.")
        print("Please check if the orthodata.csv file exists and contains valid data.")
        print("Continuing with minimal word set from command line arguments...")
        
        # Ask user to provide some words from command line if possible
        import sys
        if len(sys.argv) > 1:
            for arg in sys.argv[1:]:
                if arg.isalpha():
                    words.append(arg.lower())
        
        # If still no words, add absolute minimum set for testing
        if not words:
            print("WARNING: Using minimal word set for testing purposes.")
            words = ["the", "that", "this", "table", "turn", "time"]
            
    return words

def train_model(model, train_loader, device, num_epochs):
    """
    Train the CBOW model
    
    Args:
        model: The CBOW model
        train_loader: DataLoader with training data
        device: Device to train on (cuda/cpu)
        num_epochs: Number of training epochs
    """
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        
        for contexts, targets in train_loader:
            contexts, targets = contexts.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(contexts)
            
            # Calculate loss
            loss = criterion(logits, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "../models/char_cbow_model.pth")
    print("Model saved to ../models/char_cbow_model.pth")

def generate_word(model, char_to_idx, idx_to_char, start_char='t', max_length=10, temperature=0.8):
    """
    Generate a word starting with the given character
    
    Args:
        model: Trained CBOW model
        char_to_idx: Dictionary mapping characters to indices
        idx_to_char: Dictionary mapping indices to characters
        start_char: Starting character for the word
        max_length: Maximum length of the generated word
        temperature: Controls randomness (lower = more deterministic)
    
    Returns:
        Generated word
    """
    model.eval()
    
    # Get the model's context size (from the model's parameters)
    # Extract context size from the first layer's input dimension
    context_size = model.hidden1.in_features // EMBEDDING_DIM
    
    # Start with the given character and context padding
    word = '^' + start_char.lower()
    # Create context with appropriate padding based on context size
    context = ['^'] * (context_size - len(word)) + list(word)
    if len(context) < context_size:
        # If context_size is 1, we only need the last character
        context = context[-context_size:]
    
    # Generate characters until we reach the end token or max length
    with torch.no_grad():
        while len(word) < max_length:
            # Get context indices
            context_ids = [char_to_idx.get(c, char_to_idx['<UNK>']) for c in context[-context_size:]]
            
            # Make sure context_ids has the correct length
            if len(context_ids) < context_size:
                # Pad with start tokens if needed
                context_ids = [char_to_idx['^']] * (context_size - len(context_ids)) + context_ids
            elif len(context_ids) > context_size:
                # Truncate if somehow too long
                context_ids = context_ids[-context_size:]
            
            # Get model prediction
            inputs = torch.tensor(context_ids, dtype=torch.long).unsqueeze(0).to(next(model.parameters()).device)
            logits = model(inputs)
            
            # Apply temperature to control randomness
            probs = torch.softmax(logits / temperature, dim=1).squeeze()
            
            # Sample next character
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_char_idx]
            
            # Stop if we reach the end token
            if next_char == '$':
                break
                
            # Add to the word and update context
            word += next_char
            context = context[1:] + [next_char]
        
    # Remove start token and return the word
    return word[1:]

def main():
    """Main function to run the word generator"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load words from CSV
    words = load_words_from_csv("../data/orthodata.csv")
    
    # Ensure we have some words
    if not words:
        print("No words loaded, using fallback word list")
        words = [
            "the", "that", "then", "there", "this", "those", "thus",
            "table", "chair", "desk", "lamp", "door", "window", 
            "tree", "turtle", "tiger", "trout", "tuna", "time", "table",
            "take", "talk", "teach", "tell", "test", "think", "throw",
            "today", "tomorrow", "tonight", "too", "top", "total", "touch"
        ]
    
    # Create vocabulary
    chars = set()
    for word in words:
        chars.update(word.lower())
    
    # Add special characters
    chars.update(['^', '$', '<UNK>'])
    
    # Create character to index mapping
    char_to_idx = {c: i for i, c in enumerate(sorted(chars))}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    
    print(f"Vocabulary size: {len(char_to_idx)}")
    
    # Create dataset and dataloader
    dataset = CharDataset(words, char_to_idx, CONTEXT_SIZE)
    
    # Check if dataset is empty
    if len(dataset) == 0:
        print("Error: No training samples could be created from the words.")
        print("This might be because all words are shorter than the context size.")
        print("Adjusting context size and retrying...")
        
        # Adjust context size if needed
        adjusted_context_size = min(CONTEXT_SIZE - 1, 2)
        if adjusted_context_size >= 1:
            print(f"Retrying with context size {adjusted_context_size}")
            dataset = CharDataset(words, char_to_idx, adjusted_context_size)
    
    # If still empty, use minimal context size
    if len(dataset) == 0:
        print("Still unable to create dataset. Using minimal context size (1)...")
        dataset = CharDataset(words, char_to_idx, 1)
    
    print(f"Created dataset with {len(dataset)} samples")
    
    # Use a batch size that works with our dataset size
    effective_batch_size = min(BATCH_SIZE, max(1, len(dataset) // 2))
    dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
    
    # Create and train the model
    effective_context_size = dataset.context_size
    model = CBOWCharModel(
        vocab_size=len(char_to_idx),
        embedding_dim=EMBEDDING_DIM,
        context_size=effective_context_size,
        hidden_dim=HIDDEN_DIM
    ).to(device)
    
    print("Training model...")
    train_model(model, dataloader, device, NUM_EPOCHS)
    
    # Generate words starting with 't'
    print("\nGenerating words starting with 't':")
    for i in range(5):
        word = generate_word(model, char_to_idx, idx_to_char, start_char='t', 
                             max_length=10, temperature=0.8)
        print(f"  {i+1}. {word}")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 