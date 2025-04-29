import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
import os

# Create necessary directories
os.makedirs('../models', exist_ok=True)
os.makedirs('../results', exist_ok=True)

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, n_class=173, p_drop=0.3):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(embedding_dim)
        
        # First layer with larger width
        self.fc1 = nn.Linear(embedding_dim, 384)
        self.bn1 = nn.BatchNorm1d(384)
        
        # Second layer
        self.fc2 = nn.Linear(384, 192)
        self.bn2 = nn.BatchNorm1d(192)
        
        # Output layer
        self.out = nn.Linear(192, n_class)
        self.p_drop = p_drop
        
        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.kaiming_normal_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Embedding
        x = self.embedding(x).squeeze(1)
        
        # Input batch normalization
        x = self.input_bn(x)
        
        # First layer
        x = F.gelu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)
        
        # Second layer
        x = F.gelu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)
        
        return self.out(x)

def load_data():
    """Load the processed data"""
    X_phon_train = np.load('../processed_data/X_phon_train.npy', allow_pickle=True)
    X_phon_test = np.load('../processed_data/X_phon_test.npy', allow_pickle=True)
    y_train = np.load('../processed_data/y_train.npy')
    y_test = np.load('../processed_data/y_test.npy')
    
    # Print data information
    print(f"Training data shape: {X_phon_train.shape}")
    print(f"Test data shape: {X_phon_test.shape}")
    print(f"Number of unique phonemes in training: {len(np.unique(X_phon_train))}")
    
    # Convert to PyTorch tensors
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    return X_phon_train, X_phon_test, y_train, y_test

def create_vocabulary(X_phon_train):
    """Create vocabulary from phoneme data"""
    # Get unique phonemes
    unique_phonemes = np.unique(X_phon_train)
    
    # Create vocabulary mapping with special tokens
    vocab = {
        '<PAD>': 0,  # Padding token
        '<UNK>': 1,  # Unknown token for OOV phonemes
    }
    
    # Add phonemes to vocabulary
    for idx, phoneme in enumerate(unique_phonemes, start=2):  # Start from 2 to reserve 0,1 for special tokens
        vocab[phoneme] = idx
    
    return vocab, len(vocab)

def encode_phonemes(X_phon, vocab):
    """Encode phonemes to indices, handling OOV phonemes"""
    return torch.LongTensor([vocab.get(phoneme, vocab['<UNK>']) for phoneme in X_phon])

def train_model(model, train_loader, val_loader, device, epochs=100):
    """Train the model and return training history"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)  # Increased label smoothing
    
    # Use RAdam optimizer with conservative learning rate
    optimizer = optim.RAdam(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Use cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the restart interval after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced max norm
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), '../models/embedding_model_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')
    
    return history

def evaluate_model(model, test_loader, device):
    """Evaluate the model and return metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    mcc = matthews_corrcoef(all_labels, all_preds)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return {
        'accuracy': accuracy,
        'mcc': mcc,
        'predictions': all_preds
    }

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../results/embedding_model_training.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    X_phon_train, X_phon_test, y_train, y_test = load_data()
    
    # Create vocabulary and encode phonemes
    vocab, vocab_size = create_vocabulary(X_phon_train)
    X_train_encoded = encode_phonemes(X_phon_train, vocab)
    X_test_encoded = encode_phonemes(X_phon_test, vocab)
    
    # Print vocabulary information
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Number of OOV phonemes in test set: {sum(1 for x in X_phon_test if x not in vocab)}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_encoded, y_train)
    test_dataset = TensorDataset(X_test_encoded, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)
    
    # Create model
    num_classes = len(np.unique(y_train.numpy()))
    model = EmbeddingModel(
        vocab_size=vocab_size,
        embedding_dim=64,
        n_class=num_classes,
        p_drop=0.3  # Increased dropout
    ).to(device)
    
    print(f"\nModel created with {num_classes} output classes")
    print(f"Vocabulary size: {vocab_size}")
    
    # Train model
    history = train_model(model, train_loader, test_loader, device)
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Matthews Correlation Coefficient: {metrics['mcc']:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save model and vocabulary
    torch.save(model.state_dict(), '../models/embedding_model.pth')
    torch.save(vocab, '../models/vocabulary.pth')
    print("\nModel and vocabulary saved to ../models/")

if __name__ == "__main__":
    main()
