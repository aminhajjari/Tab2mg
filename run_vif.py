import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset, random_split
from torch.utils.data.sampler import Sampler
import torchvision
from torchvision import datasets, transforms
import itertools
import argparse
import os

# Argument parser
parser = argparse.ArgumentParser(description="Welcome to Table2Image")
parser.add_argument('--csv', type=str, required=True, help='Path to the dataset (csv)')
parser.add_argument('--save_dir', type=str, required=True, help='Path to save the final model')
parser.add_argument('--dataset_root', type=str, default='/project/def-arashmoh/shahab33/Msc/datasets', 
                    help='Root directory for image datasets (FashionMNIST and MNIST)')
args = parser.parse_args()

# Parameters
EPOCH = 50
BATCH_SIZE = 64

csv_path = args.csv
file_name = os.path.basename(csv_path).replace('.csv', '')
saving_path = args.save_dir

# Ensure saving directory exists
os.makedirs(os.path.dirname(saving_path), exist_ok=True)

# Load and preprocess the tabular data
df = pd.read_csv(csv_path)

# Print dataset info for debugging
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values per column:\n{df.isnull().sum()}")

target_col_candidates = ['target', 'class', 'outcome', 'Class', 'binaryClass', 'status', 'Target', 'TR', 'speaker', 'Home/Away', 'Outcome', 'Leaving_Certificate', 'technology', 'signal', 'label', 'Label', 'click', 'percent_pell_grant', 'Survival']
target_col = next((col for col in df.columns if col.lower() in [c.lower() for c in target_col_candidates]), None)

if target_col == None:
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    target_col = df.columns[-1]  # For printing purposes
else:
    y = df.loc[:, target_col].values
    X = df.drop(target_col, axis=1).values

print(f"Using '{target_col}' as target column")
print(f"Original data shape - X: {X.shape}, y: {y.shape}")

# Handle missing values in target column
# Option 1: Remove rows with NaN in target
nan_mask = pd.isna(y)
if nan_mask.any():
    print(f"Found {nan_mask.sum()} NaN values in target column. Removing these rows...")
    X = X[~nan_mask]
    y = y[~nan_mask]
    print(f"After removing NaN rows - X: {X.shape}, y: {y.shape}")

# Handle missing values in features
# Option 1: Remove rows with any NaN in features (you can also use imputation)
if np.isnan(X).any():
    print(f"Found NaN values in features. Handling them...")
    # Convert to DataFrame for easier handling
    X_df = pd.DataFrame(X)
    
    # Option 1a: Remove rows with any NaN
    # nan_rows = X_df.isnull().any(axis=1)
    # X = X_df[~nan_rows].values
    # y = y[~nan_rows]
    
    # Option 1b: Fill NaN with column mean (for numerical features)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    print(f"Filled NaN values with column means")
    print(f"After handling NaN in features - X: {X.shape}, y: {y.shape}")

# Ensure y has no NaN values before mapping
if pd.isna(y).any():
    raise ValueError("Target column still contains NaN values after cleaning!")

# Mapping labels for classes
unique_values = sorted(set(y))
print(f"Unique target values: {unique_values}")
print(f"Number of classes: {len(unique_values)}")

num_classes = int(len(unique_values))
value_map = {unique_values[i]: i for i in range(len(unique_values))}
y = [value_map[val] for val in y]
y = np.array(y)

n_cont_features = X.shape[1]
tab_latent_size = n_cont_features + 4
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load FashionMNIST with specified root directory
# PyTorch will look for FashionMNIST subfolder and extract .gz files if needed
fashionmnist_dataset = datasets.FashionMNIST(
    root=args.dataset_root,
    train=True,
    download=True,  # Set to True to allow extraction of .gz files
    transform=transforms.ToTensor()
)

# Load MNIST with specified root directory
# PyTorch will look for MNIST subfolder and extract .gz files if needed
mnist_dataset = datasets.MNIST(
    root=args.dataset_root,
    train=True,
    download=True,  # Set to True to allow extraction of .gz files
    transform=transforms.ToTensor()
)

# Target + 10 (MNIST)
class ModifiedLabelDataset(Dataset):
    def __init__(self, dataset, label_offset=10):
        self.dataset = dataset
        self.label_offset = label_offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label + self.label_offset

modified_mnist_dataset = ModifiedLabelDataset(mnist_dataset, label_offset=10)

# Normalize tabular features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split tabular data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create TensorDatasets
train_tabular_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_tabular_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

# Calculate number of samples needed for each label
train_tabular_label_counts = torch.bincount(train_tabular_dataset.tensors[1], minlength=int(len(unique_values)))
test_tabular_label_counts = torch.bincount(test_tabular_dataset.tensors[1], minlength=int(len(unique_values)))

num_samples_needed = train_tabular_label_counts.tolist()
num_samples_needed_test = test_tabular_label_counts.tolist()

valid_labels = {i for i in range(len(unique_values))}

# Filter FashionMNIST dataset
filtered_fashion = Subset(fashionmnist_dataset, 
                          [i for i, (_, label) in enumerate(fashionmnist_dataset) if label in valid_labels])

# Filter MNIST dataset and remap labels
filtered_mnist = Subset(modified_mnist_dataset, 
                        [i for i, (_, label) in enumerate(modified_mnist_dataset) if label in valid_labels])

# Combine FashionMNIST and MNIST
combined_dataset = ConcatDataset([filtered_fashion, filtered_mnist])

# Integrity check
indices_by_label = {label: [] for label in range(int(len(unique_values)))}

for i, (_, label) in enumerate(combined_dataset):
    if label not in indices_by_label:
        print(f"Unexpected label {label} at index {i}")
    indices_by_label[label].append(i)

# Generate repeated indices for balanced dataset
repeated_indices = {
    label: list(itertools.islice(itertools.cycle(indices_by_label[label]),
                                 num_samples_needed[label] + num_samples_needed_test[label]))
    for label in indices_by_label
}

# Align the train and test indices for both tabular and image datasets by label
aligned_train_indices = []
aligned_test_indices = []

for label in valid_labels:
    train_tab_indices = [i for i, lbl in enumerate(y_train) if lbl == label]
    test_tab_indices = [i for i, lbl in enumerate(y_test) if lbl == label]

    train_img_indices = repeated_indices[label][:num_samples_needed[label]]
    test_img_indices = repeated_indices[label][num_samples_needed[label]:num_samples_needed[label] + num_samples_needed_test[label]]

    if len(train_tab_indices) == len(train_img_indices) and len(test_tab_indices) == len(test_img_indices):
        aligned_train_indices.extend(list(zip(train_tab_indices, train_img_indices)))
        aligned_test_indices.extend(list(zip(test_tab_indices, test_img_indices)))
    else:
        raise ValueError(f"Mismatch in train/test counts for label {label}")

# Create final filtered subsets with aligned indices
train_filtered_tab_set = Subset(train_tabular_dataset, [idx[0] for idx in aligned_train_indices])
train_filtered_img_set = Subset(combined_dataset, [idx[1] for idx in aligned_train_indices])

test_filtered_tab_set = Subset(test_tabular_dataset, [idx[0] for idx in aligned_test_indices])
test_filtered_img_set = Subset(combined_dataset, [idx[1] for idx in aligned_test_indices])

# Define synchronized dataset class with consistency check
class SynchronizedDataset(Dataset):
    def __init__(self, tabular_dataset, image_dataset):
        self.tabular_dataset = tabular_dataset
        self.image_dataset = image_dataset
        assert len(self.tabular_dataset) == len(self.image_dataset), "Datasets must have the same length."

    def __len__(self):
        return len(self.tabular_dataset)

    def __getitem__(self, index):
        tab_data, tab_label = self.tabular_dataset[index]
        img_data, img_label = self.image_dataset[index]
        assert tab_label == img_label, f"Label mismatch: tab_label={tab_label}, img_label={img_label}"
        return tab_data, tab_label, img_data, img_label

# Create synchronized datasets
train_synchronized_dataset = SynchronizedDataset(train_filtered_tab_set, train_filtered_img_set)
test_synchronized_dataset = SynchronizedDataset(test_filtered_tab_set, test_filtered_img_set)

# Create data loaders
train_synchronized_loader = DataLoader(dataset=train_synchronized_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_synchronized_loader = DataLoader(dataset=test_synchronized_dataset, batch_size=BATCH_SIZE)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

d_embedding = 24

class SimpleMLP(nn.Module):
    def __init__(self, tab_latent_size=tab_latent_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(n_cont_features, tab_latent_size)
        self.fc2 = nn.Linear(tab_latent_size, int(len(unique_values)))
        self.relu = nn.ReLU()

    def forward(self, x):
        tab_latent = self.relu(self.fc1(x))
        x = self.fc2(tab_latent)
        return tab_latent, torch.sigmoid(x)

model_with_embeddings = SimpleMLP(tab_latent_size)

# VIF Embedding
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df):
    """Calculate VIF with error handling for singular matrices"""
    df = pd.DataFrame(df)
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_values = []
    
    for i in range(df.shape[1]):
        try:
            vif = variance_inflation_factor(df.values, i)
            # Handle infinite or very large VIF values
            if np.isinf(vif) or vif > 1000:
                vif = 1000  # Cap at 1000
            vif_values.append(vif)
        except:
            # If VIF calculation fails, use default value
            vif_values.append(1.0)
            
    vif_data["VIF"] = vif_values
    return vif_data

class VIFInitialization(nn.Module):
    def __init__(self, input_dim, vif_values):
        super(VIFInitialization, self).__init__()
        self.input_dim = input_dim
        self.vif_values = vif_values
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, input_dim + 4)
        self.fc2 = nn.Linear(input_dim + 4, input_dim)
        self.initialize_weights()

    def initialize_weights(self):
        with torch.no_grad():
            vif_tensor = torch.tensor(self.vif_values, dtype=torch.float32)
            # Add small epsilon to avoid division by zero
            inv_vif = 1 / (vif_tensor + 1e-8)
            for i in range(self.input_dim):
                if i < len(inv_vif):
                    self.fc1.weight.data[i, :] = inv_vif[i] / (self.input_dim + 4)
            nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x

class CVAEWithTabEmbedding(nn.Module):
    def __init__(self, tab_latent_size=8, latent_size=8):
        super(CVAEWithTabEmbedding, self).__init__()
        self.mlp = model_with_embeddings
        self.encoder = nn.Sequential(
            nn.Linear(28*28 + tab_latent_size + n_cont_features, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + tab_latent_size + n_cont_features, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
        self.final_classifier = SimpleCNN(num_classes=int(len(unique_values)))

    def encode(self, x, tab_embedding, vif_embedding):
        return self.encoder(torch.cat([x, tab_embedding, vif_embedding], dim=1))
    
    def decode(self, z, tab_embedding, vif_embedding):
        return self.decoder(torch.cat([z, tab_embedding, vif_embedding], dim=1))
    
    def forward(self, x, tab_data):
        vif_df = calculate_vif(tab_data.detach().cpu().numpy())
        vif_values = vif_df['VIF'].values
        input_dim = tab_data.shape[1]
        vif_model = VIFInitialization(input_dim, vif_values).to(DEVICE)
        vif_embedding = vif_model(tab_data)
        
        tab_embedding, tab_pred = self.mlp(tab_data)
        z = self.encode(x, tab_embedding, vif_embedding)
        recon_x = self.decode(z, tab_embedding, vif_embedding)
        img_pred = self.final_classifier(recon_x.view(-1, 1, 28, 28))
        return recon_x, tab_pred, img_pred

cvae = CVAEWithTabEmbedding(tab_latent_size).to(DEVICE)
optimizer = optim.AdamW(cvae.parameters(), lr=0.001)

def loss_function(recon_x, x, tab_pred, tab_labels, img_pred, img_labels):
    BCE = F.mse_loss(recon_x, x)
    tab_loss = F.cross_entropy(tab_pred, tab_labels)
    img_loss = F.cross_entropy(img_pred, img_labels)
    return BCE + tab_loss + img_loss

def train(model, train_data_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    num_classes = int(len(unique_values))
    
    for tab_data, tab_label, img_data, img_label in train_data_loader:
        img_data = img_data.view(-1, 28*28).to(DEVICE)
        tab_data = tab_data.to(DEVICE)
        img_label = img_label.to(DEVICE)
        tab_label = tab_label.to(DEVICE)
        
        optimizer.zero_grad()
        
        random_array = np.random.rand(img_data.shape[0], 28*28)
        x_rand = torch.Tensor(random_array).to(DEVICE)
        
        recon_x, tab_pred, img_pred = model(x_rand, tab_data)

        loss = loss_function(recon_x, img_data, tab_pred, tab_label, img_pred, img_label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f"Epoch [{epoch}/{EPOCH}], Loss: {train_loss/len(train_data_loader):.4f}")

from sklearn.metrics import roc_auc_score

def test(model, test_data_loader, epoch, best_accuracy, best_auc, best_epoch, best_model_path='best_model.pth'):
    model.eval()
    test_loss = 0
    correct_tab_total = 0
    correct_img_total = 0
    total = 0
    num_classes = int(len(unique_values))
    correct_tab = {i: 0 for i in range(num_classes)}
    total_tab = {i: 0 for i in range(num_classes)}
    correct_img = {i: 0 for i in range(num_classes)}
    total_img = {i: 0 for i in range(num_classes)}
    
    all_tab_labels = []
    all_tab_preds_proba = []
    all_img_labels = []
    all_img_preds_proba = []

    with torch.no_grad():
        for tab_data, tab_label, img_data, img_label in test_data_loader:
            img_data = img_data.view(-1, 28*28).to(DEVICE)
            tab_data = tab_data.to(DEVICE)
            img_label = img_label.to(DEVICE)
            tab_label = tab_label.to(DEVICE)
            
            random_array = np.random.rand(img_data.shape[0], 28*28)
            x_rand = torch.Tensor(random_array).view(-1, 28*28).to(DEVICE)
            
            recon_x, tab_pred, img_pred = model(x_rand, tab_data)
            
            test_loss += loss_function(recon_x, img_data, tab_pred, tab_label, img_pred, img_label).item()
            
            # Store labels and probabilities for AUC calculation
            all_tab_labels.extend(tab_label.cpu().numpy())
            all_img_labels.extend(img_label.cpu().numpy())
            
            # Get probabilities for AUC
            if num_classes > 2:
                all_tab_preds_proba.extend(F.softmax(tab_pred, dim=1).cpu().numpy())
                all_img_preds_proba.extend(F.softmax(img_pred, dim=1).cpu().numpy())
            else:
                # For binary classification
                all_tab_preds_proba.extend(torch.sigmoid(tab_pred).cpu().numpy())
                all_img_preds_proba.extend(torch.sigmoid(img_pred).cpu().numpy())

            # Get predictions
            if tab_pred.dim() == 1 or (tab_pred.dim() == 2 and tab_pred.shape[1] == 1):
                tab_predicted = (torch.sigmoid(tab_pred.squeeze()) > 0.5).long()
            else:
                tab_predicted = torch.argmax(tab_pred, dim=1)
            
            if img_pred.dim() == 1 or (img_pred.dim() == 2 and img_pred.shape[1] == 1):
                img_predicted = (torch.sigmoid(img_pred.squeeze()) > 0.5).long()
            else:
                img_predicted = torch.argmax(img_pred, dim=1)
            
            # Calculate per-class accuracy
            for i in range(len(tab_label)):
                label = tab_label[i].item()
                correct_tab[label] += (tab_predicted[i] == label).item()
                total_tab[label] += 1
                
            for i in range(len(img_label)):
                label = img_label[i].item()
                correct_img[label] += (img_predicted[i] == label).item()
                total_img[label] += 1
                
            correct_tab_total += (tab_predicted == tab_label).sum().item()
            correct_img_total += (img_predicted == img_label).sum().item()
            total += tab_label.size(0)
    
    test_loss /= len(test_data_loader)
    tab_accuracy_total = 100 * correct_tab_total / total
    img_accuracy_total = 100 * correct_img_total / total
    tab_accuracy = {cls: (correct_tab[cls] / total_tab[cls]) * 100 if total_tab[cls] > 0 else 0 for cls in range(num_classes)}
    img_accuracy = {cls: (correct_img[cls] / total_img[cls]) * 100 if total_img[cls] > 0 else 0 for cls in range(num_classes)}

    # Calculate AUC
    try:
        if num_classes > 2:
            tab_auc = roc_auc_score(all_tab_labels, all_tab_preds_proba, multi_class="ovr", average="macro")
            img_auc = roc_auc_score(all_img_labels, all_img_preds_proba, multi_class="ovr", average="macro")
        else:
            # Binary classification
            tab_auc = roc_auc_score(all_tab_labels, all_tab_preds_proba)
            img_auc = roc_auc_score(all_img_labels, all_img_preds_proba)
    except:
        print("Could not calculate AUC - possibly only one class in test set")
        tab_auc = 0
        img_auc = 0

    print(f"Epoch [{epoch}/{EPOCH}]")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Tab Accuracy: {tab_accuracy_total:.2f}%, Tab AUC: {tab_auc:.4f}")
    print(f"Img Accuracy: {img_accuracy_total:.2f}%, Img AUC: {img_auc:.4f}")
    
    if img_accuracy_total > best_accuracy:
        best_accuracy = img_accuracy_total
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with accuracy: {best_accuracy:.2f}%")
        
    if img_auc > best_auc:
        best_auc = img_auc

    return best_accuracy, best_auc, best_epoch

# Training loop
best_accuracy = 0
best_auc = 0
best_epoch = 0

print("\nStarting training...")
print(f"Number of epochs: {EPOCH}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Device: {DEVICE}")
print(f"Number of classes: {num_classes}")
print(f"Number of features: {n_cont_features}")
print("-" * 50)

for epoch in range(1, EPOCH + 1):
    train(cvae, train_synchronized_loader, optimizer, epoch)
    best_accuracy, best_auc, best_epoch = test(cvae, test_synchronized_loader, epoch, best_accuracy, best_auc, best_epoch, best_model_path=saving_path)

print("\n" + "=" * 50)
print(f'Training completed!')
print(f'Best model image classification accuracy: {best_accuracy:.4f} at epoch: {best_epoch}')
print(f'Best AUC: {best_auc:.4f}')
print(f'Model saved to: {saving_path}')
print("=" * 50)
