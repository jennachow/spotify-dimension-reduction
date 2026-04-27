import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader

# ============================================================================
# KEY OPTIMIZATION 1: Load and process data in chunks to reduce memory usage
# ============================================================================

print("Loading and preprocessing data...")
data = pd.read_csv('dataset.csv')
data = data.loc[:, ['valence', 'danceability', 'energy', 'loudness', 'speechiness', 
                     'acousticness', 'instrumentalness', 'liveness', 'tempo', 'track_genre']]
data = data.sample(n=20000)

X = data.iloc[:, 0:8]
y = data.loc[:, ['track_genre']]

# Fit encoder on full data, but convert efficiently
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
y_encoded = ohe.transform(y)
num_classes = y_encoded.shape[1]
y = np.argmax(y_encoded, axis=1)

# Convert to tensors with appropriate dtype to reduce memory
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

print(f"Dataset size: {len(X)} samples")
print(f"Memory usage: X={X.element_size() * X.nelement() / 1e6:.1f}MB, y={y.element_size() * y.nelement() / 1e6:.1f}MB")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)

# ============================================================================
# KEY OPTIMIZATION 2: Use PyTorch DataLoader for efficient batch handling
# ============================================================================

batch_size = 128  # Increased from 5 for better throughput
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ============================================================================
# Model Definition
# ============================================================================

class Multiclass(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.hidden = nn.Linear(8, 8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x

# ============================================================================
# KEY OPTIMIZATION 3: Move model to CPU explicitly (or GPU if available)
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = Multiclass(num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================================================================
# Training
# ============================================================================

n_epochs = 200
best_acc = -np.inf
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

print("\nStarting training...")
for epoch in range(n_epochs):
    epoch_loss = []
    epoch_acc = []
    
    # Training phase
    model.train()
    with tqdm.tqdm(train_loader, unit="batch", desc=f"Epoch {epoch}", leave=False) as bar:
        for X_batch, y_batch in bar:
            # Move batch to device (CPU or GPU)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            acc = (torch.argmax(y_pred, 1) == y_batch).float().mean()
            epoch_loss.append(float(loss))
            epoch_acc.append(float(acc))
            
            bar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.3f}")
    
    # ========================================================================
    # KEY OPTIMIZATION 4: Evaluate on test set in batches (not all at once)
    # ========================================================================
    
    model.eval()
    test_losses = []
    test_accs = []
    
    with torch.no_grad():  # Disable gradient computation for inference
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            acc = (torch.argmax(y_pred, 1) == y_batch).float().mean()
            test_losses.append(float(loss))
            test_accs.append(float(acc))
    
    # Average metrics
    train_loss_hist.append(np.mean(epoch_loss))
    train_acc_hist.append(np.mean(epoch_acc))
    test_loss_hist.append(np.mean(test_losses))
    test_acc_hist.append(np.mean(test_accs))
    
    avg_test_acc = test_acc_hist[-1]
    if avg_test_acc > best_acc:
        best_acc = avg_test_acc
        best_weights = copy.deepcopy(model.state_dict())
    
    if (epoch + 1) % 10 == 0:  # Print less frequently to reduce I/O
        print(f"Epoch {epoch} | Train Loss: {train_loss_hist[-1]:.4f} | Test Loss: {test_loss_hist[-1]:.4f} | Test Acc: {avg_test_acc*100:.1f}%")

print("Training complete!")

# ============================================================================
# KEY OPTIMIZATION 5: Clear CUDA cache if using GPU
# ============================================================================

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Restore best model
model.load_state_dict(best_weights)

# ============================================================================
# Plotting
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_loss_hist, label="train", linewidth=2)
ax1.plot(test_loss_hist, label="test", linewidth=2)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Cross Entropy Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(train_acc_hist, label="train", linewidth=2)
ax2.plot(test_acc_hist, label="test", linewidth=2)
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('accuracy.png') 