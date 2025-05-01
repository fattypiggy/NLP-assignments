import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader # , WeightedRandomSampler
from sklearn.metrics import matthews_corrcoef, classification_report
import matplotlib.pyplot as plt
import os
from collections import Counter
from imblearn.over_sampling import SMOTE

# Create necessary directories
os.makedirs('../models', exist_ok=True)
os.makedirs('../results', exist_ok=True)

"""
# Commenting out special loss functions
class FocalLoss(nn.Module):
    '''
    Focal Loss for dealing with class imbalance
    
    Focal Loss was proposed in:
    "Focal Loss for Dense Object Detection" by Lin et al.
    https://arxiv.org/abs/1708.02002
    
    It down-weights well-classified examples and focuses on hard examples.
    '''
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        '''
        Args:
            alpha: Weight for different classes (can be None, a float value, or a list)
            gamma: Focusing parameter. Higher gamma means stronger focus on hard examples.
            reduction: 'mean', 'sum' or 'none'
        '''
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        '''
        Args:
            inputs: Model predictions (before softmax), shape [batch_size, num_classes]
            targets: Ground truth labels, shape [batch_size]
        '''
        # Apply log_softmax to get log probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Get the log probability for the target class
        target_log_probs = log_probs.gather(1, targets.view(-1, 1))
        
        # Calculate the probability for the target class
        probs = torch.exp(log_probs)
        target_probs = probs.gather(1, targets.view(-1, 1))
        
        # Calculate the focal weight
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Add alpha weight if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # Same alpha for all classes
                alpha_t = self.alpha * torch.ones_like(target_probs)
            else:
                # Alpha is a list or tensor with different weights per class
                alpha_t = torch.tensor([self.alpha[t] for t in targets], device=inputs.device).view(-1, 1)
            
            focal_weight = alpha_t * focal_weight
        
        # Calculate focal loss
        focal_loss = -focal_weight * target_log_probs
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
            
class MixedLoss(nn.Module):
    '''
    Mixed loss combining focal loss and cross entropy with label smoothing
    '''
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, focal_weight=0.7, ce_weight=0.3):
        super(MixedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        
    def forward(self, inputs, targets):
        fl = self.focal_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        return self.focal_weight * fl + self.ce_weight * ce
"""

class FeatureModel(nn.Module):
    def __init__(self, in_dim=37, n_class=245, p_drop=0.4, hidden_dim=512):
        """
        Initialize the feature-based model
        
        Args:
            in_dim: Number of input features (phonological features)
            n_class: Number of output classes (Family_Glottocodes)
            p_drop: Dropout probability
            hidden_dim: Base size for hidden layers
        """
        super().__init__()

        # Input normalization
        self.input_bn = nn.BatchNorm1d(in_dim)
        
        # First block
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # Deeper network with residual connections
        self.fc2a = nn.Linear(hidden_dim, hidden_dim)
        self.bn2a = nn.BatchNorm1d(hidden_dim)
        self.fc2b = nn.Linear(hidden_dim, hidden_dim)
        self.bn2b = nn.BatchNorm1d(hidden_dim)
        
        # Third block with residual connection
        self.fc3a = nn.Linear(hidden_dim, hidden_dim)
        self.bn3a = nn.BatchNorm1d(hidden_dim)
        self.fc3b = nn.Linear(hidden_dim, hidden_dim)
        self.bn3b = nn.BatchNorm1d(hidden_dim)
        
        # Fourth block
        self.fc4 = nn.Linear(hidden_dim, hidden_dim//2)
        self.bn4 = nn.BatchNorm1d(hidden_dim//2)
        
        # Fifth block
        self.fc5 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.bn5 = nn.BatchNorm1d(hidden_dim//4)
        
        # Output layer
        self.out = nn.Linear(hidden_dim//4, n_class)
        
        # Dropout
        self.p_drop = p_drop
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Normalize input
        x = x.float()  # Ensure float type
        x = self.input_bn(x)

        # First block
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)
        
        # Second block (residual)
        residual = x
        x = F.relu(self.bn2a(self.fc2a(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)
        x = self.bn2b(self.fc2b(x))
        x = F.relu(x + residual)  # Add residual connection
        x = F.dropout(x, p=self.p_drop, training=self.training)
        
        # Third block (residual)
        residual = x
        x = F.relu(self.bn3a(self.fc3a(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)
        x = self.bn3b(self.fc3b(x))
        x = F.relu(x + residual)  # Add residual connection
        x = F.dropout(x, p=self.p_drop, training=self.training)
        
        # Fourth block
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)
        
        # Fifth block
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.dropout(x, p=self.p_drop, training=self.training)
        
        # Output
        return self.out(x)

def load_data():
    """Load the processed data based on Family_Glottocode"""
    X_train = np.load('../processed_data/X_feat_train.npy')
    X_test = np.load('../processed_data/X_feat_test.npy')
    y_train = np.load('../processed_data/y_train.npy')
    y_test = np.load('../processed_data/y_test.npy')
    
    # Print label information
    print(f"Label range in training set: {y_train.min()} to {y_train.max()}")
    print(f"Number of unique labels in training set: {len(np.unique(y_train))}")
    print(f"Label range in test set: {y_test.min()} to {y_test.max()}")
    print(f"Number of unique labels in test set: {len(np.unique(y_test))}")
    
    # Print class distribution
    train_counts = Counter(y_train)
    print("\nClass distribution in training set:")
    print(f"Min class count: {min(train_counts.values())}")
    print(f"Max class count: {max(train_counts.values())}")
    print(f"Imbalance ratio: {max(train_counts.values())/min(train_counts.values()):.2f}")
    
    # Apply SMOTE for classes with more than 5 samples
    print("\nApplying SMOTE oversampling technique to balance data...")
    
    # Determine which classes have enough samples for SMOTE
    min_samples_needed = 5  # SMOTE requires at least k+1 samples, default k=5
    eligible_classes = [cls for cls, count in train_counts.items() if count >= min_samples_needed]
    ineligible_classes = [cls for cls, count in train_counts.items() if count < min_samples_needed]
    
    print(f"There are {len(eligible_classes)} classes eligible for SMOTE (samples >= {min_samples_needed})")
    print(f"There are {len(ineligible_classes)} classes with too few samples for direct SMOTE application")
    
    # Analyze class distribution
    small_classes = [cls for cls, count in train_counts.items() if count < 50]
    medium_classes = [cls for cls, count in train_counts.items() if 50 <= count < 500]
    large_classes = [cls for cls, count in train_counts.items() if count >= 500]
    
    print(f"Small classes (<50 samples): {len(small_classes)}")
    print(f"Medium classes (50-500 samples): {len(medium_classes)}")
    print(f"Large classes (>500 samples): {len(large_classes)}")
    
    try:
        # Create SMOTE strategy:
        # 1. For small classes (<50 samples): Increase to 50 samples
        # 2. For medium classes (50-500 samples): Keep unchanged
        # 3. For large classes (>500 samples): Downsample to at most 50% of original, but not less than 500
        
        # First handle small classes' upsampling
        if small_classes:
            target_small = 50  # Target sample count for small classes
            
            # Only apply SMOTE to small classes
            small_sampling_strategy = {cls: target_small for cls in small_classes 
                                     if cls in eligible_classes and train_counts[cls] < target_small}
            
            if small_sampling_strategy:
                print(f"Applying SMOTE to enhance {len(small_sampling_strategy)} small classes...")
                
                # Create a subset containing only small classes and some medium/large classes for SMOTE
                # This avoids the "target samples fewer than original samples" error
                small_classes_set = set(small_classes)
                mask_small_subset = np.array([label in small_classes_set for label in y_train])
                
                # Add some medium/large class samples for SMOTE's nearest neighbor calculation
                medium_large_samples = []
                for cls in medium_classes + large_classes:
                    # Randomly select samples from each medium/large class
                    cls_indices = np.where(y_train == cls)[0]
                    samples_to_include = min(50, len(cls_indices))  # At most 50 samples per class
                    if samples_to_include > 0:
                        selected_indices = np.random.choice(cls_indices, samples_to_include, replace=False)
                        for idx in selected_indices:
                            mask_small_subset[idx] = True
                
                # Extract subset data
                X_small_subset = X_train[mask_small_subset]
                y_small_subset = y_train[mask_small_subset]
                
                # Apply SMOTE to subset
                k_neighbors = min(4, min([train_counts[cls] for cls in small_sampling_strategy]) - 1)
                k_neighbors = max(1, k_neighbors)  # Ensure k is at least 1
                
                smote = SMOTE(sampling_strategy=small_sampling_strategy, 
                              k_neighbors=k_neighbors, 
                              random_state=42)
                
                X_small_resampled, y_small_resampled = smote.fit_resample(X_small_subset, y_small_subset)
                
                # Add back samples not included in the subset
                X_rest = X_train[~mask_small_subset]
                y_rest = y_train[~mask_small_subset]
                
                X_train = np.vstack([X_small_resampled, X_rest])
                y_train = np.concatenate([y_small_resampled, y_rest])
                
                print(f"SMOTE completed: Training set size after small class enhancement: {X_train.shape[0]} samples")
        
        # Update class counts
        train_counts = Counter(y_train)
    
    except Exception as e:
        print(f"SMOTE application failed: {str(e)}")
        print("Continuing with original data...")
    
    # Check new class distribution
    new_train_counts = Counter(y_train)
    print("\nClass distribution after SMOTE:")
    print(f"Min class count: {min(new_train_counts.values())}")
    print(f"Max class count: {max(new_train_counts.values())}")
    print(f"Imbalance ratio: {max(new_train_counts.values())/min(new_train_counts.values()):.2f}")
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    return X_train, X_test, y_train, y_test

"""
# Commenting out class weight computation
def compute_class_weights(y_train):
    '''
    Compute class weights based on class frequency in the training set
    
    Args:
        y_train: Training labels tensor
        
    Returns:
        class_weights: Tensor of weights for each class
    '''
    # Convert to numpy for easier counting
    y_np = y_train.numpy()
    
    # Count samples per class
    class_counts = np.bincount(y_np)
    
    # Compute inverse frequency weighting
    n_samples = len(y_np)
    n_classes = len(class_counts)
    
    # Handle potential missing classes (sparse labels)
    effective_classes = np.where(class_counts > 0)[0]
    
    # Apply effective number of samples weighting (better than inverse freq)
    # See "Class-Balanced Loss Based on Effective Number of Samples"
    beta = 0.9999  # Smoothing factor
    effective_number = (1.0 - beta ** class_counts) / (1.0 - beta)
    class_weights = np.zeros(n_classes)
    
    for c in effective_classes:
        if effective_number[c] > 0:
            class_weights[c] = 1.0 / effective_number[c]
        else:
            class_weights[c] = 1.0
    
    # Cap weights at 15.0 to prevent extremely large weights
    class_weights = np.clip(class_weights, 0, 15.0)
    
    # Normalize weights
    class_weights = class_weights / np.sum(class_weights) * n_classes
    
    # Convert to torch tensor
    class_weights = torch.FloatTensor(class_weights)
    
    return class_weights

def create_weighted_sampler(y_train):
    '''
    Create a weighted sampler to balance class representation in each batch
    
    Args:
        y_train: Training labels tensor
        
    Returns:
        sampler: WeightedRandomSampler for balanced batches
    '''
    # Calculate sample weights based on class distribution
    class_counts = Counter(y_train.numpy())
    num_samples = len(y_train)
    
    # Create sample weights (more weight for minority classes)
    sample_weights = torch.zeros(num_samples)
    for idx, y in enumerate(y_train):
        class_id = y.item()
        count = class_counts[class_id]
        weight = 1.0 / count
        sample_weights[idx] = weight
    
    # Create and return a sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )
    
    return sampler
"""

def train_model(model, train_loader, val_loader, device, epochs=150):
    """Train the model and return training history"""
    # Calculate class weights to further handle class imbalance after SMOTE
    y_train_array = np.concatenate([y.cpu().numpy() for _, y in train_loader])
    class_counts = np.bincount(y_train_array)
    n_classes = len(class_counts)
    
    # Use square root of inverse frequency as weight
    class_weights = np.zeros(n_classes)
    for c in range(n_classes):
        if class_counts[c] > 0:
            # Use square root of inverse frequency, milder than simple inverse frequency
            class_weights[c] = 1.0 / np.sqrt(class_counts[c])
        else:
            class_weights[c] = 1.0
    
    # Normalize weights
    class_weights = class_weights / np.sum(class_weights) * n_classes
    
    # Limit weight range to prevent instability from too large weights
    class_weights = np.clip(class_weights, 0, 5.0)
    
    # Convert to torch tensor
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print("\nClass weight range:", class_weights.min().item(), "to", class_weights.max().item())
    
    # Use weighted cross entropy loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("Using cross entropy loss with class weights and SMOTE-enhanced data")
    
    # Use Adam optimizer with moderate learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
    
    # Use a more appropriate learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
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
    patience = 20  # Early stopping patience value
    
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
            
            # Gradient clipping to prevent gradient explosion
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
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Check if need to save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"Epoch {epoch+1}: New best validation loss {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 30:
                print(f"Early stopping: No improvement in validation loss for {patience} epochs")
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
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, '../models/feature_model_best.pth')
        print(f"Loaded and saved best performing model (validation loss: {best_val_loss:.4f})")
    
    return history

def evaluate_model(model, test_loader, device, num_classes, label_encoder=None):
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
    
    # Calculate per-class accuracy for the smallest and largest classes
    class_correct = {}
    class_total = {}
    for i, label in enumerate(all_labels):
        if label not in class_correct:
            class_correct[label] = 0
            class_total[label] = 0
        
        class_total[label] += 1
        if all_preds[i] == label:
            class_correct[label] += 1
    
    # Get accuracies by class
    class_accuracies = {k: class_correct[k]/class_total[k] if class_total[k] > 0 else 0 
                       for k in class_total}
    
    # Calculate average accuracy for different class sizes
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
    
    # Calculate macro average accuracy by class size
    small_acc = np.mean(small_classes) if small_classes else 0
    medium_acc = np.mean(medium_classes) if medium_classes else 0
    large_acc = np.mean(large_classes) if large_classes else 0
    
    print("\nAverage accuracy by class size:")
    print(f"Small classes (<10 samples): {small_acc:.4f} (count: {len(small_classes)})")
    print(f"Medium classes (10-50 samples): {medium_acc:.4f} (count: {len(medium_classes)})")
    print(f"Large classes (>50 samples): {large_acc:.4f} (count: {len(large_classes)})")
    
    # Print performance on a few classes
    sorted_classes = sorted(class_total.items(), key=lambda x: x[1])
    smallest_classes = sorted_classes[:5]  # 5 smallest classes
    largest_classes = sorted_classes[-5:]  # 5 largest classes
    
    print("\nPerformance on smallest classes:")
    for cls, count in smallest_classes:
        cls_name = f"{cls}"
        if label_encoder is not None and cls < len(label_encoder):
            cls_name = f"{cls} ({label_encoder[cls]})"
        print(f"Class {cls_name} (count: {count}): Accuracy = {class_accuracies[cls]:.4f}")
    
    print("\nPerformance on largest classes:")
    for cls, count in largest_classes:
        cls_name = f"{cls}"
        if label_encoder is not None and cls < len(label_encoder):
            cls_name = f"{cls} ({label_encoder[cls]})"
        print(f"Class {cls_name} (count: {count}): Accuracy = {class_accuracies[cls]:.4f}")
    
    # Generate classification report
    try:
        print("\nClassification Report:")
        report = classification_report(all_labels, all_preds, labels=range(num_classes), output_dict=True)
        
        # Print summary metrics
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
    """Plot training and validation metrics"""
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
    plt.savefig('../results/feature_model_training.png')
    plt.close()

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Load label encoder classes if available
    try:
        label_classes = np.load('../processed_data/label_classes.npy', allow_pickle=True)
        print(f"Loaded {len(label_classes)} label classes")
    except:
        label_classes = None
        print("Could not load label classes")
    
    # Verify label ranges
    print("\nVerifying label ranges:")
    print(f"Training set unique labels: {np.unique(y_train.numpy())}")
    print(f"Test set unique labels: {np.unique(y_test.numpy())}")
    print(f"Number of classes (Family_Glottocodes): {len(np.unique(y_train.numpy()))}")
    
    # Create data loaders with appropriate batch size
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Use regular DataLoader with reasonable batch size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=128,  # Smaller batch size to increase gradient estimation variance
        shuffle=True
    )
    
    # Regular DataLoader for validation/testing
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Create model with correct number of Family_Glottocodes
    num_classes = int(y_train.max().item()) + 1
    hidden_dim = 512  # Reduce network capacity to mitigate overfitting
    
    model = FeatureModel(
        in_dim=X_train.shape[1],
        n_class=num_classes,
        p_drop=0.4,  # Moderate dropout to prevent overfitting
        hidden_dim=hidden_dim
    ).to(device)
    
    print(f"\nModel created with {num_classes} output classes (Family_Glottocodes)")
    print(f"Input dimension: {X_train.shape[1]}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Using SMOTE data augmentation and early stopping")
    
    # Train model
    history = train_model(model, train_loader, test_loader, device, epochs=150)
    
    # Load best model for evaluation
    best_model_path = '../models/feature_model_best.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    
    # Evaluate model
    print("\nEvaluating best model on test set...")
    metrics = evaluate_model(model, test_loader, device, num_classes, label_classes)
    
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
