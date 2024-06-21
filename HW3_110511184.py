import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from tqdm import tqdm

# Define file paths
train_file_path = 'training.csv'
test_file_path = 'test_X.csv'
submission_file_path = 'submission.csv'

# Read data
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Feature and target variables
X_train = train_data.drop(columns=['lettr']).values
X_test = test_data.values

# Standardize data and add random noise
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) + np.random.normal(0, 0.1, X_train.shape)
X_test = scaler.transform(X_test)

# Custom Dataset class
class LetterDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

# Residual Block definition
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features, in_features)
        )

    def forward(self, x):
        return x + self.block(x)

# Autoencoder model definition
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(X_train.shape[1], 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            ResidualBlock(64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.GELU(),
            ResidualBlock(512),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, X_train.shape[1]),
            ResidualBlock(X_train.shape[1])
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# K-fold Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
early_stopping_patience = 120

num_epochs = 800
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f'Fold {fold + 1}')

    train_subset = LetterDataset(X_train[train_idx])
    val_subset = LetterDataset(X_train[val_idx])

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Training model
    scaler = torch.cuda.amp.GradScaler()
    best_loss = float('inf')
    early_stopping_counter = 0

    with tqdm(total=num_epochs, desc=f"Training Fold {fold+1}") as pbar:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for X_batch in train_loader:
                X_batch = X_batch.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs, X_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

            average_loss = running_loss / len(train_loader)

            # Validate model
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch in val_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, X_batch)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            pbar.set_postfix({'Train Loss': average_loss, 'Val Loss': val_loss})
            pbar.update(1)

            scheduler.step()

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f'best_autoencoder_weights_fold{fold+1}.pth')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping")
                    break

# Load the best model and predict on the test set
model.load_state_dict(torch.load('best_autoencoder_weights_fold1.pth'))
model.to(device)
model.eval()
test_dataset = LetterDataset(X_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
reconstruction_errors = []

with torch.no_grad():
    for X_batch in tqdm(test_loader, desc="Predicting"):
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        batch_errors = ((outputs - X_batch) ** 2).mean(dim=1)
        reconstruction_errors.extend(batch_errors.cpu().numpy())

# Save predictions
submission = pd.DataFrame({'id': range(len(reconstruction_errors)), 'outliers': reconstruction_errors})
submission.to_csv(submission_file_path, index=False)

print("Submission saved successfully.")

# Additional methods

# Method 1 - OneClass SVM
print("Training OneClass SVM...")
oc_svm = OneClassSVM(kernel='rbf', gamma='scale')
oc_svm.fit(X_train)
oc_svm_scores = -oc_svm.decision_function(X_test)

# Method 2 - KMeans
print("Training KMeans...")
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(X_train)
kmeans_distances = kmeans.transform(X_test).min(axis=1)

# Combine results
print("Combining results...")
combined_scores = (reconstruction_errors + oc_svm_scores + kmeans_distances) / 3

# Save combined predictions
submission_combined = pd.DataFrame({'id': range(len(combined_scores)), 'outliers': combined_scores})
submission_combined.to_csv('submission_combined.csv', index=False)

print("Combined submission saved successfully.")
