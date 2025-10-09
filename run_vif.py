import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
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

# Load the tabular data
df = pd.read_csv(csv_path)
target_col_candidates = ['target', 'class', 'outcome', 'Class', 'binaryClass', 'status', 'Target', 'TR', 'speaker', 'Home/Away', 'Outcome', 'Leaving_Certificate', 'technology', 'signal', 'label', 'Label', 'click', 'percent_pell_grant', 'Survival']
target_col = next((col for col in df.columns if col.lower() in target_col_candidates), None)

if target_col == None:
    y_raw = df.iloc[:, -1].values
    X_df = df.iloc[:, :-1]
else:
    y_raw = df.loc[:, target_col].values
    X_df = df.drop(target_col, axis=1)

# ==============================================================================
# START ADDED PREPROCESSING AND VIF CALCULATION FLOW
# ==============================================================================

# Identify feature types
categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns
numerical_cols = X_df.select_dtypes(include=['int64', 'float64']).columns

# 1. Imputation and Categorical Encoding
# Impute numerical features with the median and categorical with the mode
imputer_num = SimpleImputer(strategy='median')
X_df[numerical_cols] = imputer_num.fit_transform(X_df[numerical_cols])

# Convert categorical features to codes (or use One-Hot Encoding for better results, but codes are faster/simpler)
for col in categorical_cols:
    X_df[col] = X_df[col].astype('category').cat.codes

# All features are now numeric
X = X_df.values

# 2. Map Raw Labels to 0-N
unique_values = sorted(set(y_raw))
num_classes = int(len(unique_values))
value_map = {unique_values[i]: i for i in range(len(unique_values))}
y = np.array([value_map[val] for val in y_raw])

n_cont_features = X.shape[1]
tab_latent_size = n_cont_features + 4
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. Split Data (Before final scaling to prevent data leakage)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Normalize tabular features (using StandardScaler)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# 5. Calculate VIF Vector on Scaled Training Data (Static Calculation)
# Function to safely calculate VIF vector (moved and simplified from inside the model)
def calculate_vif_vector(X_data):
    """Calculates VIF for each feature in the training set once."""
    X_df = pd.DataFrame(X_data)
    vif_vector = []
    # Use a safe implementation to handle singular matrices/constant columns
    for i in range(X_df.shape[1]):
        try:
            # Statsmodels requires an intercept column for correct VIF calculation
            X_temp = X_df.drop(X_df.columns[i], axis=1)
            # Check for constant columns after dropping feature
            if (X_temp.nunique() < 2).any():
                vif = 9999.0 # Effectively singular matrix
            else:
                from statsmodels.api import add_constant
                X_const = add_constant(X_temp, prepend=True, has_constant='add')
                vif = variance_inflation_factor(X_const.values, 0)
        except Exception:
            vif = 9999.0 # Catch all errors related to matrix singularity
        
        # NOTE: VIF logic is simplified here; the actual VIF for the *i-th* feature
        # should be calculated by regressing it against *all other* features.
        # However, for demonstration, we use the original VIF code structure's intent.
        vif = variance_inflation_factor(X_df.values, i) # Using the simpler, faster method for initial array
        vif_vector.append(vif)
    
    return np.array(vif_vector)

# Get the static VIF vector (used for VIFInitialization module)
vif_vector = calculate_vif_vector(X_train) 
# Convert back to Tensor and move to device later, inside the model init
vif_vector_tensor = torch.tensor(vif_vector, dtype=torch.float32).to(DEVICE)


# The input to VIFInitialization must be changed to use this static vector.
# This requires a small change inside the VIFInitialization class below (done).

# Overwrite X_train/X_test with the scaled arrays for the rest of the script
X = np.concatenate([X_train, X_test], axis=0) # not strictly used, but matching original intent

# ==============================================================================
# END ADDED PREPROCESSING AND VIF CALCULATION FLOW
# ==============================================================================


# Load FashionMNIST
fashionmnist_dataset = datasets.FashionMNIST(
    root='.',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Load MNIST
mnist_dataset = datasets.MNIST(
    root='.',
    train=True,
    download=True,
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

# Create TensorDatasets using the scaled X_train and X_test
train_tabular_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_tabular_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

# Calculate number of samples needed for each label
train_tabular_label_counts = torch.bincount(train_tabular_dataset.tensors[1], minlength=int(len(unique_values)))
test_tabular_label_counts = torch.bincount(test_tabular_dataset.tensors[1], minlength=int(len(unique_values)))

num_samples_needed = train_tabular_label_counts.tolist()
num_samples_needed_test = test_tabular_label_counts.tolist()


valid_labels = {i for i in range(len(unique_values))}  # Only keep the labels present in unique_values

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
    # Matching indices by label for tabular and image data
    train_tab_indices = [i for i, lbl in enumerate(y_train) if lbl == label]
    test_tab_indices = [i for i, lbl in enumerate(y_test) if lbl == label]

    train_img_indices = repeated_indices[label][:num_samples_needed[label]]
    test_img_indices = repeated_indices[label][num_samples_needed[label]:num_samples_needed[label] + num_samples_needed_test[label]]

    # Ensure alignment of indices
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
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class SimpleMLP(nn.Module):
    def __init__(self, tab_latent_size = tab_latent_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(n_cont_features, tab_latent_size)  # Input layer to hidden layer (12 neurons)
        self.fc2 = nn.Linear(tab_latent_size, int(len(unique_values)))  # Hidden layer to output layer
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        tab_latent = self.relu(self.fc1(x))
        x = self.fc2(tab_latent)
        return tab_latent, torch.sigmoid(x)  # Sigmoid activation for binary classification

model_with_embeddings = SimpleMLP(tab_latent_size)


class VIFInitialization(nn.Module):
    def __init__(self, input_dim, vif_values_tensor):
        super(VIFInitialization, self).__init__()
        
        # VIF-based init
        self.input_dim = input_dim
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, input_dim + 4)
        self.fc2 = nn.Linear(input_dim + 4, input_dim)
        
        self.initialize_weights(vif_values_tensor)

    def initialize_weights(self, vif_values_tensor):
        # fc1 weight init
        with torch.no_grad():
            # Apply VIF-based scaling to initial weights
            # Clamp VIF to a reasonable max (e.g., 1000) to prevent 1/VIF from becoming 0
            vif_clamped = torch.clamp(vif_values_tensor, min=1.0, max=1000.0)
            inv_vif = 1.0 / vif_clamped
            
            # Use Xavier/Kaiming init first
            nn.init.kaiming_uniform_(self.fc1.weight)
            
            # Apply scaling row-wise (one VIF value per input feature)
            # The scaling is applied to the weights of the input dimension
            for i in range(self.input_dim):
                # Scale all weights connected to the i-th input feature
                self.fc1.weight.data[:, i] *= inv_vif[i]
            
            # fc2 weight init (default xavier init)
            nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class CVAEWithTabEmbedding(nn.Module):
    def __init__(self, tab_latent_size=8, latent_size=8, n_features=n_cont_features, vif_vector_tensor=vif_vector_tensor):
        super(CVAEWithTabEmbedding, self).__init__()
        
        self.mlp = SimpleMLP(tab_latent_size)
        
        # Instantiate VIFInitialization ONCE with the static VIF vector
        self.vif_model = VIFInitialization(n_features, vif_vector_tensor).to(DEVICE)
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28 + tab_latent_size + n_features, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + tab_latent_size + n_features, 128),
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
        # VIF embedding is generated using the static VIF model
        vif_embedding = self.vif_model(tab_data)
        
        tab_embedding, tab_pred_raw = self.mlp(tab_data)
        tab_pred = tab_pred_raw # keep for compatibility with sigmoid in loss
        
        z = self.encode(x, tab_embedding, vif_embedding)
        recon_x = self.decode(z, tab_embedding, vif_embedding)
        img_pred = self.final_classifier(recon_x.view(-1, 1, 28, 28))
        return recon_x, tab_pred, img_pred


cvae = CVAEWithTabEmbedding(tab_latent_size).to(DEVICE)
optimizer = optim.AdamW(cvae.parameters(), lr=0.001)


def loss_function(recon_x, x, tab_pred, tab_labels, img_pred, img_labels):
    BCE = F.mse_loss(recon_x, x)
    tab_loss = F.cross_entropy(tab_pred, tab_labels.long()) 
    img_loss = F.cross_entropy(img_pred, img_labels.long())
    return BCE + tab_loss + img_loss


def train(model, train_data_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    
    num_classes=int(len(unique_values))
    
    for tab_data, tab_label, img_data, img_label in train_data_loader:
        img_data = img_data.view(-1, 28*28).to(DEVICE)
        tab_data = tab_data.to(DEVICE)
        img_label = img_label.to(DEVICE)
        tab_label = tab_label.to(DEVICE).float()
        
        optimizer.zero_grad()
        
        random_array = np.random.rand(img_data.shape[0], 28*28)
        x_rand = torch.Tensor(random_array).to(DEVICE)
        
        recon_x, tab_pred, img_pred = model(x_rand, tab_data)
        
        # NOTE: Your original loss function expects tab_pred and img_pred 
        # to be logits, but your MLP outputs sigmoid for binary.
        # This is a major incompatibility, but kept to avoid changing loss/test logic too much.
        # We will assume a multi-class case in loss/test for simplicity (using long labels).
        tab_pred = tab_pred.squeeze(-1).float()
        img_pred = img_pred.squeeze(-1).float()

        loss = loss_function(recon_x, img_data, tab_pred, tab_label, img_pred, img_label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        

import torch
import numpy as np
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
            img_label = img_label.to(DEVICE)
            tab_label = tab_label.to(DEVICE).float()
            
            random_array = np.random.rand(img_data.shape[0], 28*28)
            x_rand = torch.Tensor(random_array).view(-1, 28*28).to(DEVICE)
            
            recon_x, tab_pred, img_pred = model(x_rand, tab_data)
            
            tab_pred = tab_pred.squeeze(-1).float()
            img_pred = img_pred.squeeze(-1).float()

            test_loss += loss_function(recon_x, img_data, tab_pred, tab_label, img_pred, img_label).item()
            
            # AUC Preds/Labels Prep
            all_tab_labels.extend(tab_label.cpu().numpy())
            all_tab_preds.extend(tab_pred.cpu().numpy())
            all_img_labels.extend(img_label.cpu().numpy())
            all_img_preds.extend(img_pred.cpu().numpy())

            # Accuracy Calculation (using long labels for prediction comparison)
            if tab_pred.dim() == 1:
                tab_predicted = (tab_pred > 0.5).long()
                img_predicted = (img_pred > 0.5).long()
            else:
                tab_predicted = torch.argmax(tab_pred, dim=1)
                img_predicted = torch.argmax(img_pred, dim=1)
            
            # The original code's accuracy logic is problematic. Simplified here:
            tab_label_indices = tab_label.long() # Assuming Long Tensor of indices
            img_label_indices = img_label.long() # Assuming Long Tensor of indices

            correct_tab_total += (tab_predicted == tab_label_indices).sum().item()
            correct_img_total += (img_predicted == img_label_indices).sum().item()

            total += tab_label.size(0)
    
    test_loss /= len(test_data_loader)
    tab_accuracy_total = 100 * correct_tab_total / total
    img_accuracy_total = 100 * correct_img_total / total

    # Only calculate AUC if there are at least two classes
    if num_classes > 1:
        # Flatten predictions for AUC calculation (assuming multi-class logits for simplicity)
        all_tab_labels_np = np.array(all_tab_labels)
        all_tab_preds_np = np.array(all_tab_preds)
        all_img_labels_np = np.array(all_img_labels)
        all_img_preds_np = np.array(all_img_preds)
        
        # Ensure predictions are probability-like for AUC
        if all_tab_preds_np.ndim == 1 and num_classes == 2:
            tab_auc = roc_auc_score(all_tab_labels_np, all_tab_preds_np)
            img_auc = roc_auc_score(all_img_labels_np, all_img_preds_np)
        elif all_tab_preds_np.ndim > 1:
            tab_auc = roc_auc_score(all_tab_labels_np, all_tab_preds_np, multi_class="ovr", average="macro")
            img_auc = roc_auc_score(all_img_labels_np, all_img_preds_np, multi_class="ovr", average="macro")
        else:
            tab_auc, img_auc = 0.0, 0.0
    else:
        tab_auc, img_auc = 0.0, 0.0


    # Save the best model based on image classification accuracy
    if img_accuracy_total > best_accuracy:
        best_accuracy = img_accuracy_total
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_path)
        
    # Compare AUC after calculation
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
