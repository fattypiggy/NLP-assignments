# src/train_mlp_pytorch.py

"""
Trains an MLP model (Model 1) for language family classification using feature vectors
with PyTorch.

1. Loads prepared data.
2. Defines PyTorch Dataset and DataLoader.
3. Defines and compiles an MLP model using nn.Module.
4. Trains the model with early stopping and checkpointing.
5. Evaluates the model on the test set.
6. Reports Accuracy and Matthews Correlation Coefficient (MCC).
7. Saves the trained model state dictionary.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import matthews_corrcoef, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder # Still needed for loading
import joblib
import os
import sys
import copy # For saving best model state

# Define file paths
PREPARED_DATA_DIR = "part1/data/prepared"
PREPARED_DATA_FILE = os.path.join(PREPARED_DATA_DIR, "prepared_data.npz")
LABEL_ENCODER_FILE = os.path.join(PREPARED_DATA_DIR, "label_encoder.joblib")
MODEL_SAVE_DIR = "part1/models"
# Save model state dict instead of the whole model for PyTorch best practice
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "mlp_model_1_pytorch_best.pth")

# Model Hyperparameters
HIDDEN_UNITS_1 = 128
HIDDEN_UNITS_2 = 64
DROPOUT_RATE = 0.4
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 10

def load_prepared_data(data_path, encoder_path):
    """Loads the prepared data arrays and label encoder."""
    print(f"Loading prepared data from {data_path}...")
    try:
        with np.load(data_path) as data:
            X_train = data['X_train'].astype(np.float32) # Ensure float32 for PyTorch
            y_train = data['y_train'].astype(np.int64)   # Ensure int64 for CrossEntropyLoss
            X_test = data['X_test'].astype(np.float32)
            y_test = data['y_test'].astype(np.int64)
        print("Data arrays loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please run src/prepare_training_data.py first.")
        return None
    except Exception as e:
        print(f"Error loading data file: {e}")
        return None

    print(f"Loading label encoder from {encoder_path}...")
    try:
        label_encoder = joblib.load(encoder_path)
        print("Label encoder loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Label encoder file not found at {encoder_path}")
        return None
    except Exception as e:
        print(f"Error loading label encoder: {e}")
        return None

    return X_train, y_train, X_test, y_test, label_encoder

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, HIDDEN_UNITS_1)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(DROPOUT_RATE)
        self.layer_2 = nn.Linear(HIDDEN_UNITS_1, HIDDEN_UNITS_2)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(DROPOUT_RATE)
        self.output_layer = nn.Linear(HIDDEN_UNITS_2, num_classes)
        # Note: Softmax is included in nn.CrossEntropyLoss, so not needed here

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu_1(x)
        x = self.dropout_1(x)
        x = self.layer_2(x)
        x = self.relu_2(x)
        x = self.dropout_2(x)
        x = self.output_layer(x)
        return x

def main():
    """Main function to load data, build, train, and evaluate the model."""
    print("--- Starting MLP Model Training (Model 1 - PyTorch) ---")

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data --- 
    prepared_data = load_prepared_data(PREPARED_DATA_FILE, LABEL_ENCODER_FILE)
    if prepared_data is None:
        sys.exit(1)
    X_train, y_train, X_test, y_test, label_encoder = prepared_data

    input_dim = X_train.shape[1]
    num_classes = len(label_encoder.classes_)

    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # --- Create PyTorch Datasets and DataLoaders --- 
    print("\nCreating PyTorch Datasets and DataLoaders...")
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # No need to shuffle test data, usually larger batch size is fine for evaluation
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2)
    print("DataLoaders created.")

    # --- Initialize Model, Loss, Optimizer --- 
    print("\nInitializing model, loss function, and optimizer...")
    model = MLP(input_dim, num_classes).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss() # Combines LogSoftmax and NLLLoss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Initialization complete.")

    # --- Training Loop --- 
    print("\nStarting model training...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train

        # --- Validation Loop --- 
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(): # Disable gradient calculation for validation
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(test_loader.dataset)
        epoch_val_acc = correct_val / total_val

        print(f"Epoch [{epoch+1}/{EPOCHS}] | ",
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | ",
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

        # --- Early Stopping & Model Checkpointing --- 
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Save the best model state based on validation loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, BEST_MODEL_SAVE_PATH)
            print(f"  -> Validation loss improved. Saving best model to {BEST_MODEL_SAVE_PATH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

    print("Model training finished.")

    # --- Evaluation --- 
    print("\n--- Evaluating Model on Test Set using Best Model State ---")

    # Load the best model state for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model state from epoch with lowest validation loss ({best_val_loss:.4f}).")
    else: # Should not happen if training ran for at least one epoch
        print("Warning: No best model state found. Evaluating with final epoch model.")

    model.eval() # Ensure model is in eval mode
    all_preds = []
    all_labels = []
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy()) # Collect predictions (move back to CPU)
            all_labels.extend(labels.cpu().numpy())   # Collect true labels (move back to CPU)

    final_test_loss = test_loss / total_test
    final_test_acc = correct_test / total_test

    print(f"Final Test Loss: {final_test_loss:.4f}")
    print(f"Final Test Accuracy: {final_test_acc:.4f}")

    # Calculate MCC
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    mcc = matthews_corrcoef(all_labels, all_preds)
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

    # Classification Report
    print("\nClassification Report:")
    target_names = label_encoder.classes_
    try:
        print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    except Exception as e:
        print(f"Could not generate full classification report (possibly due to label issues): {e}")
        print("Classification Report (using integer labels):")
        print(classification_report(all_labels, all_preds, zero_division=0))

    print("\n--- MLP Model Training Complete (PyTorch) ---")

if __name__ == "__main__":
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    main() 