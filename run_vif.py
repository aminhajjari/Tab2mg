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

from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
# Argument parser
parser = argparse.ArgumentParser(description="Welcome to Table2Image")
parser.add_argument('--csv', type=str, required=True, help='Path to the dataset (csv)')
parser.add_argument('--save_dir', type=str, required=True, help='Path to save the final model')
args = parser.parse_args()

# Parameters
EPOCH = 50
BATCH_SIZE = 64

csv_path = args.csv
file_name = os.path.basename(csv_path).replace('.csv', '')
saving_path = args.save_dir + '.pt'

# --- Load and preprocess the tabular data ---
df = pd.read_csv(csv_path)

# Automatically detect the target column
target_col_candidates = [
    'target', 'class', 'outcome', 'Class', 'binaryClass', 'status', 'Target',
    'TR', 'speaker', 'Home/Away', 'Outcome', 'Leaving_Certificate', 'technology',
    'signal', 'label', 'Label', 'click', 'percent_pell_grant', 'Survival',
    'diagnosis'
]

target_col = next((col for col in df.columns if col in target_col_candidates), None)

if target_col is None:
    target_col = df.columns[-1]
    print(f"[INFO] Using last column '{target_col}' as target.")

# Handle non-numeric target labels
if df[target_col].dtype == 'object' or not np.issubdtype(df[target_col].dtype, np.number):
    print(f"[INFO] Converting string labels in '{target_col}' to integers...")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df[target_col].astype(str))
    unique_values = le.classes_.tolist()
else:
    y = df[target_col].astype(int).values
    unique_values = sorted(set(y))

num_classes = len(unique_values)
print(f"[INFO] Detected {num_classes} unique classes: {unique_values}")

# Drop target to get features
X = df.drop(columns=[target_col]).values

# Ensure all features are numeric
X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(0).values

# Mapping labels for classes
unique_values = sorted(set(y))
num_classes = int(len(unique_values))
value_map = {unique_values[i]: i for i in range(len(unique_values))}
y = [value_map[val] for val in y]
y = np.array(y)

n_cont_features = X.shape[1]
tab_latent_size = n_cont_features + 4
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load FashionMNIST
DATASET_ROOT = '/project/def-arashmoh/shahab33/Msc/datasets'

fashionmnist_dataset = datasets.FashionMNIST(
    root=DATASET_ROOT,
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

mnist_dataset = datasets.MNIST(
    root=DATASET_ROOT,
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

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

# ========== FIX 1: Calculate VIF once BEFORE model creation ==========
print("[INFO] Calculating VIF values...")


def calculate_vif_safe(X_data):
    """Calculate VIF with proper error handling"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    df = pd.DataFrame(X_data)
    n_features = df.shape[1]
    vif_values = []
    
    # Suppress the warning that's causing the error message
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        for i in range(n_features):
            try:
                vif = variance_inflation_factor(df.values, i)
                # Handle invalid values
                if np.isnan(vif) or np.isinf(vif):
                    vif = 1.0
            except:
                vif = 1.0
            vif_values.append(vif)
    
    vif_values = np.array(vif_values)
    # Clip to reasonable range
    vif_values = np.clip(vif_values, 1.0, 100.0)
    return vif_values

# Use a sample for VIF calculation (faster)
# Use a sample for VIF calculation (faster)
X_sample = X_train[:min(1000, len(X_train))]  
vif_values = calculate_vif_safe(X_sample)  # ✅ Direct assignment - it's already a numpy array
print("✅ VIF values calculated once and fixed for training.")
# ========== THAT'S IT! ==========
# ========== END FIX 1 ==========

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

train_synchronized_dataset = SynchronizedDataset(train_filtered_tab_set, train_filtered_img_set)
test_synchronized_dataset = SynchronizedDataset(test_filtered_tab_set, test_filtered_img_set)

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

class SimpleMLP(nn.Module):
    def __init__(self, tab_latent_size=tab_latent_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(n_cont_features, tab_latent_size)
        self.fc2 = nn.Linear(tab_latent_size, int(len(unique_values)))
        self.relu = nn.ReLU()

    def forward(self, x):
        tab_latent = self.relu(self.fc1(x))
        x = self.fc2(tab_latent)
        return tab_latent, x

model_with_embeddings = SimpleMLP(tab_latent_size)

# ========== FIX 2: VIFInitialization with better error handling ==========
class VIFInitialization(nn.Module):
    def __init__(self, input_dim, vif_values):
        super(VIFInitialization, self).__init__()
        self.input_dim = input_dim
        self.vif_values = vif_values
        self.fc1 = nn.Linear(input_dim, input_dim + 4)
        self.fc2 = nn.Linear(input_dim + 4, input_dim)

        # ---- Normalize and invert VIF ----
        vif_tensor = torch.tensor(vif_values, dtype=torch.float32)
        vif_tensor = vif_tensor / (vif_tensor.mean() + 1e-6)
        inv_vif = 1.0 / torch.clamp(vif_tensor, min=1.0)

        # ---- Initialize weights based on inverse VIF ----
        with torch.no_grad():
            for i in range(self.fc1.weight.data.shape[0]):
                self.fc1.weight.data[i, :] = inv_vif[i % len(inv_vif)] / (self.input_dim + 4)
        print("✅ VIFInitialization: weights set using inverse VIF values.")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# ========== FIX 3: Modified CVAEWithTabEmbedding to use pre-calculated VIF ==========
class CVAEWithTabEmbedding(nn.Module):
    def __init__(self, tab_latent_size=8, latent_size=8, vif_values=None):
        super(CVAEWithTabEmbedding, self).__init__()
        
        self.mlp = model_with_embeddings
        
        # Create VIF model once during initialization
        if vif_values is not None:
            self.vif_model = VIFInitialization(n_cont_features, vif_values)
        else:
            self.vif_model = None
        
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
        # Use pre-calculated VIF model (no recalculation!)
        if self.vif_model is not None:
            vif_embedding = self.vif_model(tab_data)
        else:
            vif_embedding = tab_data
        
        tab_embedding, tab_pred = self.mlp(tab_data)
        z = self.encode(x, tab_embedding, vif_embedding)
        recon_x = self.decode(z, tab_embedding, vif_embedding)
        img_pred = self.final_classifier(recon_x.view(-1, 1, 28, 28))
        return recon_x, tab_pred, img_pred
# ========== END FIX 3 ==========

# ========== FIX 4: Create model with pre-calculated VIF ==========
#cvae = CVAEWithTabEmbedding(tab_latent_size, vif_values=vif_values).to(DEVICE)
cvae = CVAEWithTabEmbedding(tab_latent_size, vif_values=vif_values).to(DEVICE)
optimizer = optim.AdamW(cvae.parameters(), lr=0.001)

def loss_function(recon_x, x, tab_pred, tab_labels, img_pred, img_labels):
    BCE = F.mse_loss(recon_x, x)
    tab_loss = F.cross_entropy(tab_pred, tab_labels)
    img_loss = F.cross_entropy(img_pred, img_labels)
    return BCE + tab_loss + img_loss

def train(model, train_data_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    
    for tab_data, tab_label, img_data, img_label in train_data_loader:
        img_data = img_data.view(-1, 28*28).to(DEVICE)
        tab_data = tab_data.to(DEVICE)
        img_label = img_label.to(DEVICE).long()
        tab_label = tab_label.to(DEVICE).long()
        
        optimizer.zero_grad()
        
        random_array = np.random.rand(img_data.shape[0], 28*28)
        x_rand = torch.Tensor(random_array).to(DEVICE)
        
        recon_x, tab_pred, img_pred = model(x_rand, tab_data)

        loss = loss_function(recon_x, img_data, tab_pred, tab_label, img_pred, img_label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

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
    all_tab_preds = []
    all_img_labels = []
    all_img_preds = []

    with torch.no_grad():
        for tab_data, tab_label, img_data, img_label in test_data_loader:
            img_data = img_data.view(-1, 28*28).to(DEVICE)
            tab_data = tab_data.to(DEVICE)
            img_label = img_label.to(DEVICE).long()
            tab_label = tab_label.to(DEVICE).long()
            
            random_array = np.random.rand(img_data.shape[0], 28*28)
            x_rand = torch.Tensor(random_array).view(-1, 28*28).to(DEVICE)
            
            recon_x, tab_pred, img_pred = model(x_rand, tab_data)
            
            test_loss += loss_function(recon_x, img_data, tab_pred, tab_label, img_pred, img_label).item()
            
            # For AUC, use softmax probabilities
            tab_probs = F.softmax(tab_pred, dim=1)
            img_probs = F.softmax(img_pred, dim=1)
            
            # Store predictions and labels for AUC calculation
            all_tab_labels.extend(tab_label.cpu().numpy())
            all_tab_preds.extend(tab_probs.cpu().numpy())
            all_img_labels.extend(img_label.cpu().numpy())
            all_img_preds.extend(img_probs.cpu().numpy())

            # Get predicted classes
            tab_predicted = torch.argmax(tab_pred, dim=1)
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

    # ========== FIX 5: CORRECTED AUC calculation - works for ANY dataset ==========
    # Convert to numpy arrays
    all_tab_preds_arr = np.array(all_tab_preds)
    all_img_preds_arr = np.array(all_img_preds)
    all_tab_labels_arr = np.array(all_tab_labels)
    all_img_labels_arr = np.array(all_img_labels)

    # Check for NaN/Inf in tabular predictions
    if np.isnan(all_tab_preds_arr).any() or np.isinf(all_tab_preds_arr).any():
        print(f"[ERROR] NaN/Inf detected in tab predictions at epoch {epoch}")
        print(f"  Sample predictions: {all_tab_preds_arr[:5]}")
        tab_auc = 0.0
    else:
        try:
            # Automatically handle binary vs multi-class based on detected num_classes
            if num_classes == 2:
                # Binary: use probability of positive class (class 1)
                tab_auc = roc_auc_score(all_tab_labels_arr, all_tab_preds_arr[:, 1])
            else:
                # Multi-class: use one-vs-rest
                tab_auc = roc_auc_score(all_tab_labels_arr, all_tab_preds_arr, 
                                       multi_class="ovr", average="macro")
        except Exception as e:
            print(f"[ERROR] Tab AUC calculation failed at epoch {epoch}")
            print(f"  Exception: {type(e).__name__}: {str(e)}")
            print(f"  Detected classes: {num_classes}")
            print(f"  Label distribution: {np.bincount(all_tab_labels_arr)}")
            print(f"  Prediction shape: {all_tab_preds_arr.shape}")
            print(f"  Sample predictions: {all_tab_preds_arr[:3]}")
            print(f"  Sample labels: {all_tab_labels_arr[:10]}")
            tab_auc = 0.0

    # Check for NaN/Inf in image predictions
    if np.isnan(all_img_preds_arr).any() or np.isinf(all_img_preds_arr).any():
        print(f"[ERROR] NaN/Inf detected in img predictions at epoch {epoch}")
        print(f"  Sample predictions: {all_img_preds_arr[:5]}")
        img_auc = 0.0
    else:
        try:
            if num_classes == 2:
                img_auc = roc_auc_score(all_img_labels_arr, all_img_preds_arr[:, 1])
            else:
                img_auc = roc_auc_score(all_img_labels_arr, all_img_preds_arr, 
                                       multi_class="ovr", average="macro")
        except Exception as e:
            print(f"[ERROR] Img AUC calculation failed at epoch {epoch}")
            print(f"  Exception: {type(e).__name__}: {str(e)}")
            print(f"  Detected classes: {num_classes}")
            print(f"  Label distribution: {np.bincount(all_img_labels_arr)}")
            print(f"  Prediction shape: {all_img_preds_arr.shape}")
            print(f"  Sample predictions: {all_img_preds_arr[:3]}")
            print(f"  Sample labels: {all_img_labels_arr[:10]}")
            img_auc = 0.0
    # ========== END FIX 5 ==========

    # Save the best model based on image classification accuracy
    if img_accuracy_total > best_accuracy:
        best_accuracy = img_accuracy_total
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_path)
        
    if img_auc > best_auc:
        best_auc = img_auc

    return best_accuracy, best_auc, best_epoch

best_accuracy = 0
best_auc = 0
best_epoch = 0

for epoch in range(1, EPOCH + 1):
    train(cvae, train_synchronized_loader, optimizer, epoch)
    best_accuracy, best_auc, best_epoch = test(cvae, test_synchronized_loader, epoch, best_accuracy, best_auc, best_epoch, best_model_path=saving_path)

print(f'Best model image classification accuracy: {best_accuracy:.4f} at epoch: {best_epoch}, Best AUC: {best_auc:.4f}')
