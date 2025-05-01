"""
Phoneme Embedding Model for Language Family Prediction
This model uses phoneme sets for each language instead of feature vectors
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import matthews_corrcoef, classification_report
import matplotlib.pyplot as plt
import os
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Create necessary directories
os.makedirs('../models', exist_ok=True)
os.makedirs('../results', exist_ok=True)

class PhonemeEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=512, n_class=245, p_drop=0.4):
        super().__init__()
        
        # Phoneme embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Language representation layer (average phoneme embeddings)
        self.input_bn = nn.BatchNorm1d(embedding_dim)
        
        # Subsequent fully connected layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Residual connections
        self.fc2a = nn.Linear(hidden_dim, hidden_dim)
        self.bn2a = nn.BatchNorm1d(hidden_dim)
        self.fc2b = nn.Linear(hidden_dim, hidden_dim)
        self.bn2b = nn.BatchNorm1d(hidden_dim)
        
        # Another residual connection
        self.fc3a = nn.Linear(hidden_dim, hidden_dim)
        self.bn3a = nn.BatchNorm1d(hidden_dim)
        self.fc3b = nn.Linear(hidden_dim, hidden_dim)
        self.bn3b = nn.BatchNorm1d(hidden_dim)
        
        # Output layers
        self.fc4 = nn.Linear(hidden_dim, hidden_dim//2)
        self.bn4 = nn.BatchNorm1d(hidden_dim//2)
        self.fc5 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.bn5 = nn.BatchNorm1d(hidden_dim//4)
        self.out = nn.Linear(hidden_dim//4, n_class)
        
        # Dropout parameter
        self.p_drop = p_drop
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
    
    def forward(self, x, x_lengths):
        """
        Forward pass
        Args:
            x: Padded tensor of phoneme IDs [batch_size, max_phonemes]
            x_lengths: Number of actual phonemes in each sample [batch_size]
        """
        # Get embeddings [batch_size, max_phonemes, embedding_dim]
        embedded = self.embedding(x)
        
        # Average phoneme embeddings for each sample to create language representation
        # First create a mask to ignore padding
        mask = torch.arange(embedded.size(1), device=x.device)[None, :] < x_lengths[:, None]
        mask = mask.unsqueeze(-1).float()  # [batch_size, max_phonemes, 1]
        
        # Apply mask and calculate average
        masked_embedded = embedded * mask
        lang_repr = masked_embedded.sum(dim=1) / x_lengths.unsqueeze(-1).float()  # [batch_size, embedding_dim]
        
        # Apply batch normalization
        x = self.input_bn(lang_repr)
        
        # First fully connected layer
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)
        
        # Second block (with residual connection)
        residual = x
        x = F.relu(self.bn2a(self.fc2a(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)
        x = self.bn2b(self.fc2b(x))
        x = F.relu(x + residual)  # Add residual
        x = F.dropout(x, p=self.p_drop, training=self.training)
        
        # Third block (with residual connection)
        residual = x
        x = F.relu(self.bn3a(self.fc3a(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)
        x = self.bn3b(self.fc3b(x))
        x = F.relu(x + residual)  # Add residual
        x = F.dropout(x, p=self.p_drop, training=self.training)
        
        # Output layers
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)
        
        return self.out(x)

def load_phoneme_data(data_path='../data/processed_data_numeric.csv', encoding='latin1'):
    """Load and process phoneme data"""
    print(f"Loading data from {data_path}...")
    
    try:
        # Only read necessary columns: Glottocode, Phoneme, Family_Glottocode
        df = pd.read_csv(data_path, encoding=encoding, 
                        usecols=['Glottocode', 'Phoneme', 'Family_Glottocode'])
        print(f"Successfully loaded data with {len(df)} rows")
    except Exception as e:
        print(f"Error reading data: {e}")
        return None, None, None, None, None
    
    # Group by Glottocode, each language corresponds to a list of phonemes
    language_phonemes = defaultdict(list)
    family_map = {}
    
    for _, row in df.iterrows():
        glottocode = row['Glottocode']
        phoneme = row['Phoneme']
        family = row['Family_Glottocode']
        
        language_phonemes[glottocode].append(phoneme)
        family_map[glottocode] = family
    
    print(f"Processed data contains phoneme information for {len(language_phonemes)} languages")
    
    # Count phonemes for each language
    phoneme_counts = {lang: len(phonemes) for lang, phonemes in language_phonemes.items()}
    print(f"Phonemes per language: min {min(phoneme_counts.values())}, max {max(phoneme_counts.values())}, avg {sum(phoneme_counts.values())/len(phoneme_counts):.1f}")
    
    # Create phoneme vocabulary
    all_phonemes = set()
    for phonemes in language_phonemes.values():
        all_phonemes.update(phonemes)
    
    print(f"Total of {len(all_phonemes)} distinct phonemes")
    
    # Create mapping from phoneme to ID
    vocab = {
        '<PAD>': 0,  # Padding token
        '<UNK>': 1,  # Unknown phoneme token
    }
    
    for idx, phoneme in enumerate(sorted(all_phonemes), start=2):
        vocab[phoneme] = idx
    
    # Assign unique ID to each language family
    unique_families = sorted(set(family_map.values()))
    family_to_id = {family: idx for idx, family in enumerate(unique_families)}
    
    print(f"Total of {len(unique_families)} different language families")
    
    # Create dataset (each sample is a list of phoneme IDs and corresponding language family ID)
    X = []  # List of phoneme ID lists
    y = []  # Language family IDs
    languages = []  # Corresponding Glottocodes
    
    for glottocode, phonemes in language_phonemes.items():
        family = family_map.get(glottocode)
        phoneme_ids = [vocab.get(p, vocab['<UNK>']) for p in phonemes]
        
        X.append(phoneme_ids)
        y.append(family_to_id[family])
        languages.append(glottocode)
    
    # Count classes
    y_counts = Counter(y)
    print("\nClass distribution:")
    print(f"Number of classes: {len(y_counts)}")
    classes_with_one_sample = sum(1 for count in y_counts.values() if count == 1)
    print(f"Number of classes with only 1 sample: {classes_with_one_sample}")
    
    # Use safer splitting strategy to avoid stratify errors
    if classes_with_one_sample > 0:
        print("Detected classes with only one sample, using random split instead of stratified sampling")
        X_train, X_test, y_train, y_test, langs_train, langs_test = train_test_split(
            X, y, languages, test_size=0.2, random_state=42
        )
    else:
        print("Using stratified sampling for data splitting")
        X_train, X_test, y_train, y_test, langs_train, langs_test = train_test_split(
            X, y, languages, test_size=0.2, random_state=42, stratify=y
        )
    
    print(f"Data split into training set ({len(X_train)} languages) and test set ({len(X_test)} languages)")
    
    # Create reverse mapping (from ID to family name)
    id_to_family = {idx: family for family, idx in family_to_id.items()}
    label_names = [id_to_family[i] for i in range(len(unique_families))]
    
    # Count label distribution
    train_counts = Counter(y_train)
    print("\nTraining set label distribution:")
    print(f"Min samples: {min(train_counts.values())}")
    print(f"Max samples: {max(train_counts.values())}")
    print(f"Imbalance ratio: {max(train_counts.values())/min(train_counts.values()):.2f}")
    
    return X_train, X_test, y_train, y_test, vocab, label_names

def collate_fn(batch):
    """Pack phoneme ID lists of different lengths into a batch"""
    # Separate features and labels
    phonemes, labels = zip(*batch)
    
    # Get length of each sequence
    lengths = torch.tensor([len(p) for p in phonemes])
    
    # Pad phoneme ID lists to equal length
    padded_phonemes = pad_sequence([torch.tensor(p) for p in phonemes], 
                                 batch_first=True, 
                                 padding_value=0)
    
    return padded_phonemes, torch.tensor(labels), lengths

def train_model(model, train_loader, val_loader, device, epochs=150):
    """Train the model and return training history"""
    # Calculate class weights to handle imbalance
    # Collect labels from all batches
    all_labels = []
    for _, labels, _ in train_loader:
        all_labels.extend(labels.numpy())
    
    y_train_array = np.array(all_labels)
    class_counts = np.bincount(y_train_array)
    n_classes = len(class_counts)
    
    # Use square root of inverse frequency as weight
    class_weights = np.zeros(n_classes)
    for c in range(n_classes):
        if class_counts[c] > 0:
            class_weights[c] = 1.0 / np.sqrt(class_counts[c])
        else:
            class_weights[c] = 1.0
    
    # Normalize
    class_weights = class_weights / np.sum(class_weights) * n_classes
    class_weights = np.clip(class_weights, 0, 5.0)  # Limit maximum weight
    
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print("\nClass weight range:", class_weights.min().item(), "to", class_weights.max().item())
    
    # Use weighted cross entropy loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("Using cross entropy loss with class weights")
    
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
    
    # Use learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    patience = 20  # Early stopping patience
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for phonemes, labels, lengths in train_loader:
            phonemes, labels, lengths = phonemes.to(device), labels.to(device), lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(phonemes, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for phonemes, labels, lengths in val_loader:
                phonemes, labels, lengths = phonemes.to(device), labels.to(device), lengths.to(device)
                outputs = model(phonemes, lengths)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"Epoch {epoch+1}: New best validation loss {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 30:
                print(f"Early stopping: Validation loss hasn't improved for {patience} epochs")
                break
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, '../models/phoneme_model_best.pth')
        print(f"Loaded and saved best performing model (validation loss: {best_val_loss:.4f})")
    
    return history

def evaluate_model(model, test_loader, device, num_classes, label_names=None):
    """Evaluate the model and return metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for phonemes, labels, lengths in test_loader:
            phonemes, labels, lengths = phonemes.to(device), labels.to(device), lengths.to(device)
            outputs = model(phonemes, lengths)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate MCC and accuracy
    mcc = matthews_corrcoef(all_labels, all_preds)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Calculate accuracy by class
    class_correct = {}
    class_total = {}
    for i, label in enumerate(all_labels):
        if label not in class_correct:
            class_correct[label] = 0
            class_total[label] = 0
        
        class_total[label] += 1
        if all_preds[i] == label:
            class_correct[label] += 1
    
    # Calculate average accuracy by class size
    class_accuracies = {k: class_correct[k]/class_total[k] if class_total[k] > 0 else 0 
                       for k in class_total}
    
    small_classes = []
    medium_classes = []
    large_classes = []
    
    for cls, count in class_total.items():
        if count < 10:
            small_classes.append(class_accuracies[cls])
        elif count < 50:
            medium_classes.append(class_accuracies[cls])
        else:
            large_classes.append(class_accuracies[cls])
    
    small_acc = np.mean(small_classes) if small_classes else 0
    medium_acc = np.mean(medium_classes) if medium_classes else 0
    large_acc = np.mean(large_classes) if large_classes else 0
    
    print("\nAverage accuracy by class size:")
    print(f"Small classes (<10 samples): {small_acc:.4f} (count: {len(small_classes)})")
    print(f"Medium classes (10-50 samples): {medium_acc:.4f} (count: {len(medium_classes)})")
    print(f"Large classes (>50 samples): {large_acc:.4f} (count: {len(large_classes)})")
    
    # Print performance on smallest and largest classes
    sorted_classes = sorted(class_total.items(), key=lambda x: x[1])
    smallest_classes = sorted_classes[:5]
    largest_classes = sorted_classes[-5:]
    
    print("\nPerformance on smallest classes:")
    for cls, count in smallest_classes:
        cls_name = f"{cls}"
        if label_names is not None and cls < len(label_names):
            cls_name = f"{cls} ({label_names[cls]})"
        print(f"Class {cls_name} (count: {count}): Accuracy = {class_accuracies[cls]:.4f}")
    
    print("\nPerformance on largest classes:")
    for cls, count in largest_classes:
        cls_name = f"{cls}"
        if label_names is not None and cls < len(label_names):
            cls_name = f"{cls} ({label_names[cls]})"
        print(f"Class {cls_name} (count: {count}): Accuracy = {class_accuracies[cls]:.4f}")
    
    # Generate classification report
    try:
        print("\nClassification Report:")
        report = classification_report(all_labels, all_preds, labels=range(num_classes), output_dict=True)
        
        print(f"Macro avg precision: {report['macro avg']['precision']:.4f}")
        print(f"Macro avg recall: {report['macro avg']['recall']:.4f}")
        print(f"Macro avg f1-score: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted avg f1-score: {report['weighted avg']['f1-score']:.4f}")
    except Exception as e:
        print(f"Could not generate classification report: {str(e)}")
    
    return {
        'accuracy': accuracy,
        'mcc': mcc,
        'predictions': all_preds,
        'class_accuracies': class_accuracies,
        'small_class_acc': small_acc,
        'medium_class_acc': medium_acc,
        'large_class_acc': large_acc
    }

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('../results/phoneme_model_training.png')
    plt.close()

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    try:
        X_train, X_test, y_train, y_test, vocab, label_names = load_phoneme_data()
    except UnicodeDecodeError:
        print("UTF-8 encoding failed, trying latin1 encoding...")
        X_train, X_test, y_train, y_test, vocab, label_names = load_phoneme_data(encoding='latin1')
    
    # Create datasets
    train_dataset = list(zip(X_train, y_train))
    test_dataset = list(zip(X_test, y_test))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model
    num_classes = max(max(y_train), max(y_test)) + 1
    vocab_size = len(vocab)
    
    model = PhonemeEmbeddingModel(
        vocab_size=vocab_size,
        embedding_dim=64,
        hidden_dim=512,
        n_class=num_classes,
        p_drop=0.4
    ).to(device)
    
    print(f"\nCreated model with {num_classes} output classes (language families)")
    print(f"Vocabulary size: {vocab_size} (distinct phonemes)")
    print(f"Embedding dimension: 64")
    print(f"Hidden dimension: 512")
    
    # Train model
    history = train_model(model, train_loader, test_loader, device, epochs=150)
    
    # Load best model
    best_model_path = '../models/phoneme_model_best.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model: {best_model_path}")
    
    # Evaluate model
    print("\nEvaluating best model...")
    metrics = evaluate_model(model, test_loader, device, num_classes, label_names)
    
    # Print results
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Matthews Correlation Coefficient: {metrics['mcc']:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    torch.save(model.state_dict(), '../models/phoneme_model.pth')
    torch.save(vocab, '../models/phoneme_vocab.pth')
    print("\nModel saved to ../models/phoneme_model.pth")
    print("Vocabulary saved to ../models/phoneme_vocab.pth")

if __name__ == "__main__":
    main() 