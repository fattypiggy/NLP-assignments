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

class FeatureModel(nn.Module):
    def __init__(self, in_dim=37, n_class=173, p_drop=0.2):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(in_dim)

        self.fc1 = nn.Linear(in_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.out = nn.Linear(128, n_class)
        self.p_drop = p_drop

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x * 2.0 - 1.0                    # 0/1 → -1/+1
        x = self.input_bn(x)

        # Linear ➜ BN ➜ RELU ➜ Dropout
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)

        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)

        return self.out(x)

def load_data():
    """Load the processed data"""
    X_train = np.load('../processed_data/X_feat_train.npy')
    X_test = np.load('../processed_data/X_feat_test.npy')
    y_train = np.load('../processed_data/y_train.npy')
    y_test = np.load('../processed_data/y_test.npy')
    
    # Print label information
    print(f"Label range in training set: {y_train.min()} to {y_train.max()}")
    print(f"Number of unique labels in training set: {len(np.unique(y_train))}")
    print(f"Label range in test set: {y_test.min()} to {y_test.max()}")
    print(f"Number of unique labels in test set: {len(np.unique(y_test))}")
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    return X_train, X_test, y_train, y_test

def train_model(model, train_loader, val_loader, device, epochs=100):
    """Train the model and return training history"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Use AdamW optimizer with higher learning rate
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    
    # Use OneCycleLR scheduler for better learning rate control
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,  # Warm up for 30% of training
        div_factor=25,  # Initial lr = max_lr/25
        final_div_factor=1000  # Final lr = max_lr/1000
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience = 15  # Increased patience
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
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
            torch.save(model.state_dict(), '../models/feature_model_best.pth')
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
    plt.savefig('../results/feature_model_training.png')
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Verify label ranges
    print("\nVerifying label ranges:")
    print(f"Training set unique labels: {np.unique(y_train.numpy())}")
    print(f"Test set unique labels: {np.unique(y_test.numpy())}")
    print(f"Number of classes: {len(np.unique(y_train.numpy()))}")
    
    # Create data loaders with larger batch size
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)
    
    # Create model
    num_classes = len(np.unique(y_train.numpy()))
    model = FeatureModel(
        in_dim=X_train.shape[1],
        n_class=num_classes,
        p_drop=0.2
    ).to(device)
    
    print(f"\nModel created with {num_classes} output classes")
    print(f"Input dimension: {X_train.shape[1]}")
    
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
    
    # Save final model
    torch.save(model.state_dict(), '../models/feature_model.pth')
    print("\nModel saved to ../models/feature_model.pth")

if __name__ == "__main__":
    main()
