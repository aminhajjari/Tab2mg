# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: PyTorch 2.2 (NGC 23.11/Python 3.10) on Backend.AI
#     language: python
#     name: python3
# ---

# +
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset
from torchvision import datasets, transforms
import argparse
import itertools
import os

from rtdl_num_embeddings import (
    compute_bins,
    PeriodicEmbeddings,
    PiecewiseLinearEncoding,
    PiecewiseLinearEmbeddings
)
from rtdl_revisiting_models import MLP

# argparse 설정
parser = argparse.ArgumentParser(description="Train a diabetes model with a CSV file.")
parser.add_argument('--csv', required=True, help="Path to the input CSV file.")
parser.add_argument('--csv_name', type=str, required=True, help='Path to the dataset (csv)')
args = parser.parse_args()

csv_file = args.csv
csv_name = args.csv_name

# 하이퍼파라미터 설정
EPOCH = 10
BATCH_SIZE = 64
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
# print("Using Device:", DEVICE)

df = pd.read_csv(csv_file)
target_col_candidates = ['target', 'class', 'outcome', 'Class', 'binaryClass', 'status', 'Target', 'TR', 'speaker', 'Home/Away', 'Outcome', 'Leaving_Certificate', 'technology', 'signal', 'label', 'Label', 'click', 'percent_pell_grant', 'Survival']
target_col = next((col for col in df.columns if col.lower() in target_col_candidates), None)
if target_col == None:
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
else:
    y = df.loc[:, target_col].values
    X = df.drop(target_col, axis=1).values


EPOCH = 50
BATCH_SIZE = 64 #512 # 256
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
# print("Using Device:", DEVICE)

# Mapping labels for classes
unique_values = sorted(set(y))
num_classes = int(len(unique_values))
value_map = {unique_values[i]: i for i in range(len(unique_values))}
y = [value_map[val] for val in y]
y = np.array(y)

n_cont_features = X.shape[1]
tab_latent_size = n_cont_features + 4
#     saving_path = '/home/work/DLmath/seungeun/tab/tab_model/' + file_name + '_default.pt'
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load FashionMNIST
fashionmnist_dataset = datasets.FashionMNIST(
    root='/home/work/DLmath/seungeun/tab/fashionmnist',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# Load MNIST
mnist_dataset = datasets.MNIST(
    root='/home/work/DLmath/seungeun/tab/mnist',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# 라벨에 10을 더하는 커스텀 데이터셋 정의
class ModifiedLabelDataset(Dataset):
    def __init__(self, dataset, label_offset=10):
        self.dataset = dataset
        self.label_offset = label_offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label + self.label_offset

# 커스텀 데이터셋 생성
modified_mnist_dataset = ModifiedLabelDataset(mnist_dataset, label_offset=10)
#######################변경 사항###################################

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

valid_labels = {i for i in range(len(unique_values))}  # Only keep the labels present in unique_values

# Filter FashionMNIST dataset
filtered_fashion = Subset(fashionmnist_dataset, 
                          [i for i, (_, label) in enumerate(fashionmnist_dataset) if label in valid_labels])


# Filter MNIST dataset and remap labels
filtered_mnist = Subset(modified_mnist_dataset, 
                        [i for i, (_, label) in enumerate(modified_mnist_dataset) if label in valid_labels])



# Combine FashionMNIST and MNIST
combined_dataset = ConcatDataset([filtered_fashion, filtered_mnist])

# 확인용 디버깅 코드
indices_by_label = {label: [] for label in range(int(len(unique_values)))}

for i, (_, label) in enumerate(combined_dataset):
    if label not in indices_by_label:
        print(f"Unexpected label {label} at index {i}")  # 잘못된 라벨 출력
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




######################################

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=1):
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


# n_cont_features = X.shape[1]
# tab_latent_size = n_cont_features + 4

tab_output_size = X.shape[1]
img_latent_size = tab_output_size + 4
# print('tab_output_size:',tab_output_size, 'img_latent_size:', img_latent_size)

class InverseCVAE(nn.Module):
    def __init__(self, img_latent_size=12, tab_output_size=8):
        super(InverseCVAE, self).__init__()

        self.img_encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, img_latent_size)
        )

        self.tabular_decoder = nn.Sequential(
            nn.Linear(img_latent_size + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, tab_output_size),  # Output tabular data
        )
        self.cnn = SimpleCNN(num_classes=num_classes)

    def encode(self, img_data):
        return self.img_encoder(img_data)

    def decode(self, z, img_latent):
        return self.tabular_decoder(torch.cat([z, img_latent], dim=1))

    def forward(self, img_data):
        img = img_data.view(-1, 28, 28).unsqueeze(1)
#         print(img.shape)
        img_latent = self.cnn(img)  # Classify the image first
        z = self.encode(img_data)
#         print('z:', z.shape, 'img_latent:', img_latent.shape)
        recon_tabular = self.decode(z, img_latent)  # Decode into tabular data
        return recon_tabular, img_latent


cvae = InverseCVAE(img_latent_size=img_latent_size, tab_output_size=tab_output_size).to(DEVICE)
optimizer = optim.Adam(cvae.parameters(), lr=0.01)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Decay LR by a factor of 0.1 every 10 epochs

def loss_function(tab_data, img_data, recon_tabular, img_latent, img_label):
    BCE = F.mse_loss(tab_data, recon_tabular)
    img_loss = F.cross_entropy(img_latent, img_label.long())
    return BCE + img_loss


def train(model, train_data_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for tab_data, tab_label, img_data, img_label in train_data_loader:
        img_data = img_data.view(-1, 28*28).to(DEVICE)
        tab_data = tab_data.to(DEVICE)
        img_label = img_label.to(DEVICE)
        tab_label = tab_label.to(DEVICE)

        optimizer.zero_grad()

        # random_array = np.random.rand(img_data.shape[0], 28*28)
        # x_rand = torch.Tensor(random_array).to(DEVICE)

        recon_tabular, img_latent = model(img_data)
        loss = loss_function(tab_data, img_data, recon_tabular, img_latent, img_label)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_data_loader):.4f}')
    #     scheduler.step()

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

def test(model, test_data_loader, epoch, best_accuracy, best_epoch, best_model_path='best_model.pth'):
    model.eval()
    test_loss = 0
    correct_tab_total = 0
    correct_img_total = 0
    total = 0
    correct_tab = {0: 0, 1: 0}  # Dictionary to store correct predictions for each class
    total_tab = {0: 0, 1: 0}    # Dictionary to store total samples for each class
    correct_img = {0: 0, 1: 0}  # Dictionary to store correct predictions for each class
    total_img = {0: 0, 1: 0}    # Dictionary to store total samples for each class

    all_img_labels = []
    all_img_preds = []

    with torch.no_grad():
        for tab_data, tab_label, img_data, img_label in test_data_loader:
            img_data = img_data.view(-1, 28*28).to(DEVICE)
            tab_data = tab_data.to(DEVICE)
            img_label = img_label.to(DEVICE)
            tab_label = tab_label.to(DEVICE)

            # random_array = np.random.rand(img_data.shape[0], 28*28)
            # x_rand = torch.Tensor(random_array).view(-1, 28*28).to(DEVICE)

            recon_tabular, img_latent = model(img_data)
            test_loss += loss_function(tab_data, img_data, recon_tabular, img_latent, img_label).item()



    test_loss /= len(test_data_loader)


    # Save the best model based on image classification accuracy
    if test_loss < best_loss:
        best_accuracy = test_loss
        best_epoch = epoch
        # torch.save(model.state_dict(), best_model_path)
        torch.save(model, best_model_path)
        print(f'====> Saving new best model with tab recon loss: {best_loss:.2f} at epoch: {best_epoch}')

    return best_accuracy, best_epoch

best_loss = 100
best_epoch = 0

for epoch in range(1, EPOCH + 1):
    train(cvae, train_synchronized_loader, optimizer, epoch)
    best_loss, best_epoch = test(cvae, test_synchronized_loader, epoch, best_loss, best_epoch, best_model_path = f"{csv_name}/{csv_name}_inv.pt")

print(f'Best model tab recon loss: {best_loss:.2f} at epoch: {best_epoch}')

