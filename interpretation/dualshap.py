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

# * 학습하기 위한 순서<br>1. 터미널에서 개인폴더의 interpretation폴더로 이동<br>2. 터미널에서 아래의 명령어 실행
# <br><br>**터미널 명령어**<br>
# seq 200 | xargs -I {} python3 Final_csv_ModelFigSave_mean_mmd_elbo_diabetes_interpretation_val_3.py {} >> Final_csv_ModelFigSave_mean_mmd_elbo_diabetes_interpretation_val_3.txt
# <br>
#
# * 학습 완료 후 models1,2폴더, csv폴더, figures폴더, txt파일을 다른 폴더로 옮기기

# +
# # !pip install rtdl_num_embeddings
# # !pip install rtdl_revisiting_models
# # !pip install shap
# # !pip install opencv-python==4.5.5.64
# # !pip install rtdl
# # !pip install torch --upgrade


# +
# # !pip install librosa==0.10.1
# # !pip install msgpack==1.0.2
# # !pip install transformers==4.33.1
# # !pip install bitsandbytes==0.43.1
# # !pip install accelerate cudart huggingface_hub torchcontrib
# # !pip install scikit-learn==1.5.1

# +
import random
import os
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
import argparse
import itertools

# from rtdl_num_embeddings import (
#     compute_bins,
#     PeriodicEmbeddings,
#     PiecewiseLinearEncoding,
#     PiecewiseLinearEmbeddings
# )
# from rtdl_revisiting_models import MLP

import shap

# %matplotlib inline

# # # Set the seed value
# seed = 2

# # Set the random seed for reproducibility
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# # # If you are using CUDA, you should also set the seed for GPU computations
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# Argument parser
parser = argparse.ArgumentParser(description="Welcome to INTERPRETABLE TAB2IMG")
parser.add_argument('--csv', type=str, required=True, help='Path to the dataset (csv)')
parser.add_argument('--csv_name', type=str, required=True, help='Path to the dataset (csv)')
parser.add_argument('--index', type=str, required=True, help='Path to the dataset (csv)')
args = parser.parse_args()

# Parameters
EPOCH = 50
BATCH_SIZE = 512 # 256

csv_path = args.csv
csv_name = args.csv_name
index = args.index

# Load and preprocess the tabular data
df = pd.read_csv(csv_path)

# 목표 column 결정
target_col_candidates = ['target', 'class', 'outcome', 'Class', 'binaryClass', 'status', 'Target', 'TR', 'speaker', 'Home/Away', 'Outcome', 'Leaving_Certificate', 'technology', 'signal', 'label', 'Label', 'click', 'percent_pell_grant', 'Survival']
target_col = next((col for col in df.columns if col.lower() in target_col_candidates), None)

# CSV 데이터 로드 및 전처리
if target_col == None:
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
else:
    # CSV 데이터 로드 및 전처리
    y = df.loc[:, target_col].values
    X = df.drop(target_col, axis=1).values

# Mapping labels for three classes
unique_values = sorted(set(y))
num_classes = int(len(unique_values))
value_map = {unique_values[i]: i for i in range(len(unique_values))}
y = [value_map[val] for val in y]
y = np.array(y) 

n_cont_features = X.shape[1]
n_samples = X.shape[-1]
tab_latent_size = n_cont_features + 4

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Load FashionMNIST
fashionmnist_dataset = datasets.FashionMNIST(
    root='/home/work/DLmath/Subin/tabular/data/fashionmnist',
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

# # Combine labels for FashionMNIST (0~9) and MNIST (10~19)
# selected_labels_fashion = list(range(10))
# selected_labels_mnist = list(range(10, 20))

# # Filter FashionMNIST dataset
# fashion_indices = [i for i, (_, label) in enumerate(fashionmnist_dataset) if label in selected_labels_fashion]
# filtered_fashion = Subset(fashionmnist_dataset, fashion_indices)

# # Filter MNIST dataset and remap labels
# mnist_indices = [i for i, (_, label) in enumerate(mnist_dataset) if label in range(10)]
# filtered_mnist = Subset(mnist_dataset, mnist_indices)



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


# -

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


########################################안 쓰이는 클래스######################################################
class SimpleCNN_seq(nn.Module):
    def __init__(self, num_classes=1):
        super(SimpleCNN_seq, self).__init__()
        
        # Convolutional layers with pooling and ReLU activation
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers with dropout and ReLU activation
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)  # Output changed to 1 for binary classification
        )


    def forward(self, x):
        x = self.conv_layers(x)  # Pass through conv layers
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)  # Pass through fully connected layers
        return x

# +
d_embedding = 24
# m_cont_embeddings = PeriodicEmbeddings(n_cont_features, lite=False)

# Compute bins
# Using quantile-based bins
# quantile_bins = compute_bins(X_train_tabular_tensor)

# # Using target-aware tree-based bins
# tree_bins = compute_bins(
#     X_train_tabular_tensor,
#     tree_kwargs={'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4},
#     y=y_train_tabular_tensor,
#     regression=True,
# )

# # Define MLP-Q / MLP-T model
# mlp_config = {
#     'd_out': tab_latent_size,  # For example, a single regression task.
#     'n_blocks': 2,
#     'd_block': 256,
#     'dropout': 0.1,
# } # 2 12 0

# model_with_embeddings = nn.Sequential(
#     m_cont_embeddings,
#     nn.Flatten(),
#     MLP(d_in=n_cont_features * d_embedding, **mlp_config)
# )


class SimpleMLP(nn.Module):
    def __init__(self, tab_latent_size = tab_latent_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(n_cont_features, tab_latent_size)  # Input layer to hidden layer (12 neurons)
        self.fc2 = nn.Linear(tab_latent_size, int(len(unique_values)))  # Hidden layer to output layer
        self.relu = nn.ReLU()        # ReLU activation function

    def forward(self, x):
        tab_latent = self.relu(self.fc1(x))
        x = self.fc2(tab_latent)
        return tab_latent, torch.sigmoid(x)  # Sigmoid activation for binary classification

model_with_embeddings = SimpleMLP(tab_latent_size)

# mlp_q_model = nn.Sequential(
#     PiecewiseLinearEncoding(quantile_bins),
#     nn.Flatten(),
#     MLP(d_in=sum(len(b) - 1 for b in quantile_bins), **mlp_config)
# )

# mlp_t_model = nn.Sequential(
#     PiecewiseLinearEncoding(tree_bins),
#     nn.Flatten(),
#     MLP(d_in=sum(len(b) - 1 for b in tree_bins), **mlp_config)
# )

# # Define MLP-QLR / MLP-TLR model
# mlp_qlr_model = nn.Sequential(
#     PiecewiseLinearEmbeddings(quantile_bins, d_embedding, activation=True),
#     nn.Flatten(),
#     MLP(d_in=n_cont_features * d_embedding, **mlp_config)
# )

# mlp_tlr_model = nn.Sequential(
#     PiecewiseLinearEmbeddings(tree_bins, d_embedding, activation=True),
#     nn.Flatten(),
#     MLP(d_in=n_cont_features * d_embedding, **mlp_config)
# )

# +
class CVAEWithTabEmbedding(nn.Module):
    def __init__(self, tab_latent_size=8, latent_size=8):
        super(CVAEWithTabEmbedding, self).__init__()
        
        self.mlp = model_with_embeddings
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28 + tab_latent_size, 128),
#             nn.Linear(tab_latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + tab_latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
        
#         self.tab_classifier = nn.Linear(tab_latent_size, int(len(unique_values)))
        self.final_classifier = SimpleCNN(num_classes=int(len(unique_values)))  # Custom ResNet-based classifier for images

    def encode(self, x, tab_embedding):
        return self.encoder(torch.cat([x, tab_embedding], dim=1))
#         return self.encoder(tab_embedding)
    
    def decode(self, z, tab_embedding):
        return self.decoder(torch.cat([z, tab_embedding], dim=1))
    
    def forward(self, x, tab_data):
        tab_embedding, tab_pred = self.mlp(tab_data)
        z = self.encode(x, tab_embedding)
        recon_x = self.decode(z, tab_embedding)
#         tab_pred = tab_embedding
#         tab_pred = self.tab_classifier(tab_embedding)
        img_pred = self.final_classifier(recon_x.view(-1, 1, 28, 28))  # Reshape recon_x for CNN input
        return recon_x, tab_pred, img_pred


# +
cvae = CVAEWithTabEmbedding(tab_latent_size).to(DEVICE)
optimizer = optim.AdamW(cvae.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Decay LR by a factor of 0.1 every 10 epochs

import torch.nn.functional as F

def loss_function(recon_x, x, tab_pred, tab_labels, img_pred, img_labels):
    BCE = F.mse_loss(recon_x, x)  # Reconstruction loss
    tab_loss = F.cross_entropy(tab_pred, tab_labels)  # Use BCE for multi-label target
    img_loss = F.cross_entropy(img_pred, img_labels)
    return BCE + tab_loss + img_loss


# -

########################################안 쓰이는 클래스(바로 아래 셀이 쓰이는 데 아래 셀이 안 쓰임)######################################################
class SimpleCNN_INV(nn.Module):
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


########################################안 쓰이는 클래스######################################################
class InverseCVAE(nn.Module):
    def __init__(self, img_latent_size=12, tab_output_size=8):
        super(InverseCVAE, self).__init__()
        
        self.img_encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, img_latent_size)
        )
        
        self.tabular_decoder = nn.Sequential(
            nn.Linear(img_latent_size + 1, 128),
            nn.ReLU(),
            nn.Linear(128, tab_output_size),  # Output tabular data
        )
        
        self.cnn = SimpleCNN_INV(num_classes=num_classes)

    def encode(self, img_data):
        return self.img_encoder(img_data)
    
    def decode(self, z, img_latent):
        return self.tabular_decoder(torch.cat([z, img_latent], dim=1))
    
    def forward(self, img_data):
        img = img_data.view(-1, 28, 28).unsqueeze(1)
        img_latent = self.cnn(img)  # Classify the image first
        z = self.encode(img_data)
        recon_tabular = self.decode(z, img_latent)  # Decode into tabular data
        return recon_tabular, img_latent


########################################안 쓰이는 클래스######################################################
class TABMLP(nn.Module):
    def __init__(self, tab_latent_size=12):
        super(TABMLP, self).__init__()
        self.fc1 = nn.Linear(8, tab_latent_size)  # 8 features as input
        self.fc2 = nn.Linear(tab_latent_size, 1)  # Output layer for binary classification
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)


# +
# P_I_X_model = torch.load('diabetes.pt').to(DEVICE)
fmnist_path = 'fashion_mnist_classification_' + str(num_classes) + '.pt'
P_I_X_model = torch.load(fmnist_path).to(DEVICE)
P_X_I_model = torch.load(f'{csv_name}/{csv_name}_inv.pt').to(DEVICE)
P_F_X_model = torch.load(f'{csv_name}/{csv_name}_tab.pt').to(DEVICE)

# 모델의 모든 파라미터에 대해 requires_grad를 False로 설정하여 freeze
# for param in P_I_X_model.parameters():
#     param.requires_grad = False

# for param in P_X_I_model.parameters():
#     param.requires_grad = False

# for param in P_F_X_model.parameters():
#     param.requires_grad = False

# +
# def P_I_X_model_rev(x):
#     return P_I_X_model_rev(x)[1]

# +
# def img_wrapper(x):
#     # NumPy -> Tensor 변환
#     x_tensor = torch.tensor(x, dtype=torch.float32)
#     x_tensor = x_tensor.view(-1, 1, 28, 28)
#     x_tensor = x_tensor.to(DEVICE)  # 필요한 경우 GPU로 이동
#     return P_I_X_model.final_classifier(x_tensor).detach().cpu().numpy()

def tab_wrapper(x):
    # NumPy -> Tensor 변환
    x_tensor = torch.tensor(x, dtype=torch.float32)
    x_tensor = x_tensor.to(DEVICE)  # 필요한 경우 GPU로 이동
    return P_F_X_model(x_tensor).detach().cpu().numpy()


# +
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import shap
import torch.nn.functional as F

def train(model, train_data_loader, optimizer, epoch):
    model.train()
    train_loss = 0
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
        
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_data_loader):.4f}')


def test(model, test_data_loader, epoch, best_accuracy, best_epoch, best_model_path='best_model.pth'):
    model.eval()
    test_loss = 0
    correct_tab_total = 0
    correct_img_total = 0
    total = 0
    num_classes = int(len(unique_values))  # Adjust this for your actual number of classes
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
            tab_label = tab_label.to(DEVICE)
            
            random_array = np.random.rand(img_data.shape[0], 28*28)
            x_rand = torch.Tensor(random_array).view(-1, 28*28).to(DEVICE)
            
            recon_x, tab_pred, img_pred = model(x_rand, tab_data)

            # Squeeze the last dimension if it's of size 1 (e.g., for binary classification)
            tab_pred = tab_pred.squeeze(-1)
            img_pred = img_pred.squeeze(-1)

            # Convert labels to float (not long, as per your request)
            tab_label = F.one_hot(tab_label.long(), num_classes=num_classes).float()
            img_label = F.one_hot(img_label.long(), num_classes=num_classes).float()
            
            test_loss += loss_function(recon_x, img_data, tab_pred, tab_label, img_pred, img_label).item()
            
            # Store predictions and labels for AUC calculation
            all_tab_labels.extend(tab_label.cpu().numpy())
            all_tab_preds.extend(tab_pred.cpu().numpy())  # Probabilities for each class
            all_img_labels.extend(img_label.cpu().numpy())
            all_img_preds.extend(img_pred.cpu().numpy())  # Probabilities for each class

            # Handle binary or multi-class cases
            if tab_pred.dim() == 1:  # Binary case (single output per sample)
                tab_predicted = (tab_pred > 0.5).long()  # Use a threshold for binary classification
            else:  # Multi-class case (probabilities for each class)
                tab_predicted = torch.argmax(tab_pred, dim=1)
            
            for i in range(len(tab_label)):
                label = torch.argmax(tab_label[i]).item()  # Convert one-hot to class index
                correct_tab[label] += (tab_predicted[i] == label).item()
                total_tab[label] += 1

            # Calculate accuracy for image classification
            if img_pred.dim() == 1:  # Binary classification for images
                img_predicted = (img_pred > 0.5).long()  # Use a threshold for binary classification
            else:  # Multi-class case (probabilities for each class)
                img_predicted = torch.argmax(img_pred, dim=1)
            
            for i in range(len(img_label)):
                label = torch.argmax(img_label[i]).item()  # Convert one-hot to class index
                correct_img[label] += (img_predicted[i] == label).item()
                total_img[label] += 1
                
            tab_label_indices = torch.argmax(tab_label, dim=1)  # Convert one-hot encoded labels back to indices
            correct_tab_total += (tab_predicted == tab_label_indices).sum().item()

            img_label_indices = torch.argmax(img_label, dim=1)
            correct_img_total += (img_predicted == img_label_indices).sum().item()

            total += tab_label.size(0)
    
    test_loss /= len(test_data_loader)
    tab_accuracy_total = 100 * correct_tab_total / total
    img_accuracy_total = 100 * correct_img_total / total
    tab_accuracy = {cls: (correct_tab[cls] / total_tab[cls]) * 100 if total_tab[cls] > 0 else 0 for cls in range(num_classes)}
    img_accuracy = {cls: (correct_img[cls] / total_img[cls]) * 100 if total_img[cls] > 0 else 0 for cls in range(num_classes)}
    
    #######################
    # Calculate AUC for tabular and image classification (multi-class)
#     img_auc = roc_auc_score(all_img_labels, np.array(all_img_preds)[:, 1], multi_class='ovr')

    print(f'====> Test set loss: {test_loss:.4f}')
    print(f'====> Test set tabular classification accuracy: {tab_accuracy_total:.2f}%')
    print(f'====> Test set image classification accuracy: {img_accuracy_total:.2f}%')
    for cls in range(num_classes):
        print(f'====> Test set tabular classification accuracy for Class {cls}: {tab_accuracy[cls]:.2f}%')
        print(f'====> Test set image classification accuracy for Class {cls}: {img_accuracy[cls]:.2f}%')
#     print(f'====> Test set image classification AUC: {img_auc:.4f}')

    # Save the best model based on image classification accuracy
    if img_accuracy_total > best_accuracy:
#         os.makedirs(f'{csv_name}/models1', exist_ok=True)
        best_accuracy = img_accuracy_total
        best_epoch = epoch
#         torch.save(model.state_dict(), best_model_path) 저장안함
        print(f'====> Saving new best model with image classification accuracy: {best_accuracy:.2f}% at epoch: {best_epoch}')
        
    P_I_X = 0
    P_X_I = 0
    P_F_X = 0
    P_F_I = 0
    
    if epoch == EPOCH:
        for tab_data, tab_label, img_data, img_label in test_data_loader:
            img_data = img_data.view(-1, 28*28).to(DEVICE)
            tab_data = tab_data.to(DEVICE)
            img_label = img_label.to(DEVICE)
            tab_label = tab_label.to(DEVICE)
#             print(img_data.shape, tab_data.shape, img_label.shape, tab_label.shape)

            random_array = np.random.rand(img_data.shape[0], 28*28)
            x_rand = torch.Tensor(random_array).view(-1, 28*28).to(DEVICE)
            
#             print(x_rand.shape, tab_data.shape)
            P_I_X, tab_pred, img_pred = model(x_rand, tab_data)
#             print(P_I_X.shape, tab_pred.shape, img_pred.shape)
            P_X_I, _ = P_X_I_model(img_data)
#             print(P_X_I.shape)

            P_I_X_img = P_I_X.view(-1, 1, 28, 28)
#             print(P_I_X_img.shape)
            image_explainer = shap.DeepExplainer(P_I_X_model, P_I_X_img)
            tabular_explainer = shap.KernelExplainer(tab_wrapper, tab_data.detach().cpu().numpy())
            P_F_X = tabular_explainer.shap_values(tab_data.detach().cpu().numpy())
#             print(P_F_X.shape)
            P_F_I = image_explainer.shap_values(P_I_X_img)
#             print(P_F_I.shape)

            np.save('I_X.npy', P_I_X_img.detach().cpu().numpy())
            np.save('X_I.npy', tab_data.detach().cpu().numpy())
            np.save('F_X', P_F_X)
            np.save('F_I.npy', P_F_I)
    return best_accuracy, best_epoch, P_I_X, P_X_I, P_F_X, P_F_I, tab_predicted



# +
import sys

best_accuracy = 0
best_epoch = 0

for epoch in range(1, EPOCH + 1):
    train(cvae, train_synchronized_loader, optimizer, epoch)
    best_accuracy, best_epoch, P_I_X, P_X_I, P_F_X, P_F_I, tab_predicted = test(cvae, test_synchronized_loader, epoch, best_accuracy, best_epoch, best_model_path=f'{csv_name}/models1/interpretation_{index}.pt')

print(f'Best model image classification accuracy: {best_accuracy:.2f}% at epoch: {best_epoch}')
# -

P_F_X_df = pd.DataFrame(P_F_X[np.arange(int(P_I_X.shape[0])), :, tab_predicted.detach().cpu().numpy()])
save_path = 'csv_P_F_X_df'
os.makedirs(os.path.join(csv_name, save_path), exist_ok=True)
P_F_X_df.to_csv(os.path.join(csv_name, os.path.join(save_path, f'P_F_X_df_{index}.csv')), index=False, header=False)

# +
# print(P_I_X.shape, P_F_X.shape, P_F_I.shape)
P_F_X = P_F_X[np.arange(int(P_I_X.shape[0])), :, tab_predicted.detach().cpu().numpy()]
P_F_I = P_F_I[np.arange(int(P_I_X.shape[0])), :,:,:, tab_predicted.detach().cpu().numpy()]

# print(P_I_X.shape, P_F_X.shape, P_F_I.shape)
P_I_X = P_I_X.view(-1, 28, 28)

# print(P_I_X.shape, P_F_X.shape, P_F_I.shape)
# -

# torch.Size([148, 784]) (148, 19, 5) (148, 1, 28, 28, 5)<br>
# torch.Size([148, 784]) (148, 19) (148, 1, 28, 28)<br>
# torch.Size([148, 28, 28]) (148, 19) (148, 1, 28, 28)<br>

# +
class CustomModel(nn.Module):
    def __init__(self, num_classes, size):
        super(CustomModel, self).__init__()
        
        self.num_classes = num_classes
        self.size = size
        
        self.conv1 = nn.Conv2d(size*4, 154, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(154, 36, kernel_size=3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(144, 72, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(72*25, size*self.num_classes)
        
    def forward(self, x): 
        first_channel = x[:, ::2, ::2] # 154, 14, 14
        second_channel = x[:, 1::2, ::2]
        third_channel = x[:, ::2, 1::2]
        fourth_channel = x[:, 1::2, 1::2]

        x_concat = torch.cat([first_channel, second_channel,
                             third_channel, fourth_channel], dim=0).unsqueeze(0)  # (1, 616, 14, 14)

        x = self.conv1(x_concat)
        x = self.conv2(x) # 1,36,14,14
        
        first_channel = x[:, :, ::2, ::2] # 36,7,7
        second_channel = x[:, :, 1::2, ::2]
        third_channel = x[:, :, ::2, 1::2] 
        fourth_channel = x[:, :, 1::2, 1::2] 

        x_concat = torch.cat([first_channel, second_channel,
                             third_channel, fourth_channel], dim=1)  # (1, 144, 7, 7)
        
        x = self.conv3(x_concat) # (1, 72, 5, 5)
        x = self.fc1(x.view(1, -1))

        return x.reshape(self.size, self.num_classes)
    
view_model = CustomModel(num_classes=P_F_X.shape[1], size=P_F_X.shape[0]).to(DEVICE)


# +
# # MMD Loss 구성하기 위한 함수 설정

# # bandwidth가 작을수록 커널 함수가 국소적인 정보에 더 민감해지고, 값이 크면 더 넓은 범위의 데이터를 유사하다고 간주
# def mmd_loss(x, y, kernel_bandwidth=0.5): # 분모 : , eps=1e-6
#     def gaussian_kernel(a, b):
#         dist = ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(2)
#         return torch.exp(-dist / (2 * kernel_bandwidth ** 2)) #  + eps
    
#     xx_kernel = gaussian_kernel(x, x)
#     yy_kernel = gaussian_kernel(y, y)
#     xy_kernel = gaussian_kernel(x, y)
    
#     mmd = xx_kernel.mean() + yy_kernel.mean() - 2 * xy_kernel.mean()

#     return mmd

# +
# MMD Loss 구성하기 위한 함수 설정

# bandwidth가 작을수록 커널 함수가 국소적인 정보에 더 민감해지고, 값이 크면 더 넓은 범위의 데이터를 유사하다고 간주
def mmd_loss(x, y): # 분모 : , eps=1e-6
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(DEVICE),
                  torch.zeros(xx.shape).to(DEVICE),
                  torch.zeros(xx.shape).to(DEVICE))

    bandwidth_range = [10, 15, 20, 50]
    for a in bandwidth_range:
        XX += torch.exp(-0.5*dxx/a)
        YY += torch.exp(-0.5*dyy/a)
        XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)


# +
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
import sys
import os

from torch.distributions import Normal, kl_divergence

class Starting_MLP_I(nn.Module):
    def __init__(self, input_size, output_size):
        super(Starting_MLP_I, self).__init__()
        self.fc_mean = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.fc_std = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softplus()  # 표준편차는 양수로 만들어야 하므로 Softplus 사용
        )
    
    def forward(self, x):
        mean = self.fc_mean(x)
        std = self.fc_std(x)
        return mean, std
    
class Starting_MLP_X(nn.Module):
    def __init__(self, input_size, output_size):
        super(Starting_MLP_X, self).__init__()
        self.fc_mean = nn.Sequential(
            nn.Linear(input_size, 12),
            nn.ReLU(),
            nn.Linear(12, output_size)
        )
        self.fc_std = nn.Sequential(
            nn.Linear(input_size, 12),
            nn.ReLU(),
            nn.Linear(12, output_size),
            nn.Softplus()  # 표준편차는 양수로 만들어야 하므로 Softplus 사용
        )
    
    def forward(self, x):
        mean = self.fc_mean(x)
        std = self.fc_std(x)
        return mean, std

# def save_model(model, optimizer, index, path="models2"): 저장안함
#     os.makedirs(os.path.join(csv_name, path), exist_ok=True)
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict()
#     }, os.path.join(csv_name, path)+f'/model2_{index}.pt')
#     #print(f"Model saved to {save_path}")

def infonce_loss(P_I_F_X, P_X_F_I):
    P_I_F_X = P_I_F_X.view(P_I_F_X.size(0), -1)
    P_X_F_I = P_X_F_I.view(P_X_F_I.size(0), -1)

    similarity = torch.matmul(P_I_F_X, P_X_F_I.t())

    pos = torch.diag(similarity)
    neg = similarity.sum(dim=1)
    loss = -(pos - neg).mean()
    return loss

def bounded_rsample(dist, mean, lower, upper, threshold_ratio=0.9, max_attempts=100):
    for _ in range(max_attempts):
        sample = dist.rsample()
        # 일정 비율 이상의 요소가 조건 만족
        valid_ratio = torch.sum((sample >= lower) & (sample <= upper)).item() / sample.numel()
        if valid_ratio >= threshold_ratio:
            return sample
    # 실패 시: 평균값 반환
    return mean

def train(P_I_X, P_X_I, P_F_X, P_F_I, min_epochs=100, lr=0.001):
    P_I_X = P_I_X.detach().to(DEVICE) + 10
    P_X_I = P_X_I.detach().to(DEVICE) + 10
    P_F_X = torch.Tensor(P_F_X).to(DEVICE) + 10
    P_F_I = torch.Tensor(P_F_I).to(DEVICE) + 10

    mlp_P_I_F_X = Starting_MLP_I(input_size=28*28*2, output_size=28*28).to(DEVICE)
    mlp_P_X_F_I = Starting_MLP_X(input_size=n_cont_features*2, output_size=n_cont_features).to(DEVICE)
    optimizer = optim.Adam(list(mlp_P_I_F_X.parameters()) + list(mlp_P_X_F_I.parameters()), lr=lr)

    best_reconstruction_loss = float('inf')
    best_P_F_I_X, best_P_F_X_I = None, None
    epoch = 0
    
    recon_weight = 1.0
    vi_weight = 0.5

    while epoch < min_epochs and best_reconstruction_loss > 0.0001:
        optimizer.zero_grad()

        mean_P_I_F_X, std_P_I_F_X = mlp_P_I_F_X(torch.cat([P_I_X.view(-1, 28*28), P_F_I.view(-1, 28*28)], dim=-1))
        mean_P_X_F_I, std_P_X_F_I = mlp_P_X_F_I(torch.cat([P_X_I, P_F_X], dim=-1))

        gaussian_I_F_X = Normal(mean_P_I_F_X, std_P_I_F_X)
        gaussian_X_F_I = Normal(mean_P_X_F_I, std_P_X_F_I)
        
        P_I_F_X_pred = bounded_rsample(gaussian_I_F_X, mean_P_I_F_X, mean_P_I_F_X - 1.645 * std_P_I_F_X / np.sqrt(n_samples), mean_P_I_F_X - 1.645 * std_P_I_F_X / np.sqrt(n_samples))
        P_X_F_I_pred = bounded_rsample(gaussian_X_F_I, mean_P_X_F_I, mean_P_X_F_I - 1.645 * std_P_X_F_I / np.sqrt(n_samples), mean_P_X_F_I - 1.645 * std_P_X_F_I / np.sqrt(n_samples))

        divide = P_I_F_X_pred.view(-1, 28, 28) / P_I_X
#         print(divide.shape, P_F_I.shape)
        P_F_I_X = view_model(divide) * P_F_X
#         print(P_F_I_X.shape)
#         P_F_I_X = (F.avg_pool2d(divide, kernel_size=4, stride=4, padding=2).mean(dim=-1)) * P_F_X
        P_F_X_I = (P_X_F_I_pred / P_X_I) * view_model(P_F_I.squeeze(1))
#         print(P_F_X_I.shape)
#         P_F_X_I = (P_X_F_I_pred / P_X_I) * F.avg_pool2d(P_F_I, kernel_size=4, stride=4, padding=2).mean(dim=-1)

        reconstruction_loss = ((P_F_I_X - P_F_X_I) ** 2).mean()
        recon_mmd_loss = mmd_loss(P_F_I_X, P_F_X_I)
        
        # KL divergence 계산
        P_F_I_X_kl = Normal(P_F_I_X.mean(), P_F_I_X.std())
        P_F_X_I_kl = Normal(P_F_X_I.mean(), P_F_X_I.std())
        kl_div = kl_divergence(P_F_I_X_kl, P_F_X_I_kl)
        
        mi_loss = infonce_loss(view_model(P_I_F_X_pred.view(-1, 28, 28)), P_X_F_I_pred)
        
        total_loss = reconstruction_loss + mi_loss + recon_mmd_loss + kl_div
        total_loss.backward()
        optimizer.step()

        if reconstruction_loss.item() < best_reconstruction_loss:
            best_reconstruction_loss = reconstruction_loss.item()
            best_P_F_I_X = P_F_I_X.detach().clone()
            best_P_F_X_I = P_F_X_I.detach().clone()
            
            # 모델 저장 (덮어쓰기) 저장안함
#             save_model(mlp_P_I_F_X, optimizer, index)
#             save_model(mlp_P_X_F_I, optimizer, index)
            
            mi_loss_return = mi_loss
            reconstruction_loss_return = reconstruction_loss
            recon_mmd_loss_return = recon_mmd_loss
            kl_div_return = kl_div
            sigma_return_1 = std_P_I_F_X
            sigma_return_2 = std_P_X_F_I

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}], Total Loss: {total_loss.item():.4f}, '
                  f'MI Loss: {mi_loss.item():.4f}, '
                  f'Recon Loss: {reconstruction_loss.item():.4f}, '
                  f'MMD Loss: {recon_mmd_loss.item():.4f}, '
                  f'kld Loss: {kl_div:.4f}'
                  f'Best Recon Loss: {best_reconstruction_loss:.4f}'
                  )

        epoch += 1

    print(f'Training finished after {epoch} epochs. Best Reconstruction Loss: {best_reconstruction_loss:.4f}')
    
    return best_P_F_I_X, best_P_F_X_I, mi_loss_return, reconstruction_loss_return, recon_mmd_loss_return, kl_div_return, sigma_return_1, sigma_return_2

P_F_I_X, P_F_X_I, mi_loss_return, reconstruction_loss_return, recon_mmd_loss_return, kl_div_return, sigma_return_1, sigma_return_2 = train(P_I_X, P_X_I, P_F_X, P_F_I, lr=0.00001)

# +
save_path = 'loss_df'
os.makedirs(os.path.join(csv_name, save_path), exist_ok=True)


# 함수 정의
def save_results_to_csv(index, mi_loss, reconstruction_loss, recon_mmd_loss, kl_div, sigma_return_1, sigma_return_2, file_path="results.csv", file_path_sigma1="results.csv", file_path_sigma2="results.csv"):
    # 저장할 데이터 생성
    data = {
        "index": [index],
        "mi_loss": [mi_loss.detach().cpu().numpy()],
        "reconstruction_loss": [reconstruction_loss.detach().cpu().numpy()],
        "recon_mmd_loss": [recon_mmd_loss.detach().cpu().numpy()],
        "kl_div": [kl_div.detach().cpu().numpy()],
    }
    df = pd.DataFrame(data)
    
    
    data_sigma1 = sigma_return_1.detach().cpu().numpy()
    data_sigma2 = sigma_return_2.detach().cpu().numpy()
    
    df_sigma1 = pd.DataFrame(data_sigma1)
    df_sigma2 = pd.DataFrame(data_sigma2)

    # 파일이 이미 존재하면 기존 데이터에 추가, 없으면 새로 생성
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)  # append mode
    else:
        df.to_csv(file_path, mode='w', header=True, index=False)  # write mode
        
    if os.path.exists(file_path_sigma1):
        df_sigma1.to_csv(file_path_sigma1, mode='a', header=False, index=False)  # append mode
    else:
        df_sigma1.to_csv(file_path_sigma1, mode='w', header=True, index=False)  # write mode
        
    if os.path.exists(file_path_sigma2):
        df_sigma2.to_csv(file_path_sigma2, mode='a', header=False, index=False)  # append mode
    else:
        df_sigma2.to_csv(file_path_sigma2, mode='w', header=True, index=False)  # write mode
        
        
save_results_to_csv(index=index, mi_loss=mi_loss_return, reconstruction_loss=reconstruction_loss_return, recon_mmd_loss=recon_mmd_loss_return, kl_div=kl_div_return, sigma_return_1=sigma_return_1, sigma_return_2=sigma_return_2, file_path=os.path.join(csv_name, os.path.join(save_path, f'loss_df.csv')), file_path_sigma1=os.path.join(csv_name, os.path.join(save_path, f'loss_df_sigma1.csv')), file_path_sigma2=os.path.join(csv_name, os.path.join(save_path, f'loss_df_sigma2.csv')))
# -

# * 모든 instance에 대한 mean 시각화 및 저장
# * 이미지 파일 저장 경로 변경

# +
import numpy as np
import matplotlib.pyplot as plt

P_F_I_X_numpy = P_F_I_X.detach().cpu().numpy()

plt.figure(figsize=(10, 6))
values = np.mean(P_F_I_X_numpy, axis=0)
plt.bar(range(P_F_I_X_numpy.shape[1]), values, color='blue')

# 값 출력하는 부분 추가
for i, value in enumerate(values):
    plt.text(i, value, str(value), ha='center', va='bottom', fontsize=8)
    
plt.xticks(ticks=range(P_F_I_X_numpy.shape[1]), labels=range(P_F_I_X_numpy.shape[1]))

plt.title(f"Mean of P_F_I_X - Bar Graph - Index {index}")
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.grid(axis='y')  # Optional: Add grid lines for better readability

# 이미지 파일로 저장
save_path = 'figures'
os.makedirs(os.path.join(csv_name, save_path), exist_ok=True)
plt.savefig(os.path.join(csv_name, os.path.join(save_path, f'plot_P_F_I_X_{index}.png')))  # 파일 이름에 인덱스를 반영

# 그래프를 화면에 표시하려면 아래의 코드를 사용
# plt.show()

# 플롯을 저장한 후 클리어
plt.close()



# +
import numpy as np
import matplotlib.pyplot as plt

P_F_X_I_numpy = P_F_X_I.detach().cpu().numpy()

plt.figure(figsize=(10, 6))
values = np.mean(P_F_X_I_numpy, axis=0)
plt.bar(range(P_F_X_I_numpy.shape[1]), values, color='blue')

# 값 출력하는 부분 추가
for i, value in enumerate(values):
    plt.text(i, value, str(value), ha='center', va='bottom', fontsize=8)

plt.xticks(ticks=range(P_F_X_I_numpy.shape[1]), labels=range(P_F_X_I_numpy.shape[1]))
    
plt.title(f"Mean of P_F_X_I - Bar Graph - Index {index}")
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.grid(axis='y')  # Optional: Add grid lines for better readability

# 이미지 파일로 저장
save_path = 'figures'
os.makedirs(os.path.join(csv_name, save_path), exist_ok=True)
plt.savefig(os.path.join(csv_name, os.path.join(save_path, f'plot_P_F_X_I_{index}.png')))  # 파일 이름에 인덱스를 반영

# 그래프를 화면에 표시하려면 아래의 코드를 사용
# plt.show()

# 플롯을 저장한 후 클리어
plt.close()


# +
import numpy as np
import matplotlib.pyplot as plt

P_F_I_X_numpy = P_F_I_X.detach().cpu().numpy()
P_F_X_I_numpy = P_F_X_I.detach().cpu().numpy()

mean_PF = (P_F_X_I_numpy + P_F_I_X_numpy)/2

plt.figure(figsize=(10, 6))
values = values = np.mean(mean_PF, axis=0)
plt.bar(range(mean_PF.shape[1]), values, color='blue')

# 값 출력하는 부분 추가
for i, value in enumerate(values):
    plt.text(i, value, str(value), ha='center', va='bottom', fontsize=8)

plt.xticks(ticks=range(mean_PF.shape[1]), labels=range(mean_PF.shape[1]))

plt.title(f"Mean of mean_PF - Bar Graph - Index {index}")
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.grid(axis='y')  # Optional: Add grid lines for better readability

# 이미지 파일로 저장
save_path = 'figures'
os.makedirs(os.path.join(csv_name, save_path), exist_ok=True)
plt.savefig(os.path.join(csv_name, os.path.join(save_path, f'plot_mean_PF_{index}.png')))  # 파일 이름에 인덱스를 반영

# 그래프를 화면에 표시하려면 아래의 코드를 사용
# plt.show()

# 플롯을 저장한 후 클리어
plt.close()
# -

# ---

# * FI뽑은 것(154, 8)에 모두 csv파일에 저장

# +
P_F_I_X_numpy = P_F_I_X.detach().cpu().numpy()
P_F_X_I_numpy = P_F_X_I.detach().cpu().numpy()
mean_PF = (P_F_X_I_numpy + P_F_I_X_numpy)/2

P_F_I_X_df = pd.DataFrame(P_F_I_X_numpy)
P_F_X_I_df = pd.DataFrame(P_F_X_I_numpy)
mean_PF_df = pd.DataFrame(mean_PF)

save_path = 'csv'
os.makedirs(os.path.join(csv_name, save_path), exist_ok=True)

P_F_I_X_df.to_csv(os.path.join(csv_name, os.path.join(save_path, f'P_F_I_X_df_{index}.csv')), index=False, header=False)
P_F_X_I_df.to_csv(os.path.join(csv_name, os.path.join(save_path, f'P_F_X_I_df_{index}.csv')), index=False, header=False)
mean_PF_df.to_csv(os.path.join(csv_name, os.path.join(save_path, f'mean_PF_df_{index}.csv')), index=False, header=False)



# +
## training부터 iteration 반복하는 코드 -> 그냥 for문 사용 안 하기로...
# -








