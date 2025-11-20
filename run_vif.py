import random
import os
import logging
import warnings
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import argparse
import itertools

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset
import torchvision
from torchvision import datasets, transforms
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============ LOGGING SETUP ============
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ============ DATA LOADING FUNCTIONS ============

def check_file_size(filepath: str, max_mb: int = 1000) -> float:
    """Check if file is too large before loading."""
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(
            f"File too large: {size_mb:.1f}MB (max: {max_mb}MB). "
            f"Consider sampling or splitting the dataset."
        )
    logger.info(f"File size: {size_mb:.1f}MB")
    return size_mb


def smart_fill_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Intelligently handle missing values."""
    initial_na = df.isna().sum().sum()
    
    df = df.dropna(axis=1, thresh=len(df) * (1 - threshold))
    df = df.dropna(how='all')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        mode_val = df[col].mode()
        fill_val = mode_val[0] if len(mode_val) > 0 else 'unknown'
        df[col] = df[col].fillna(fill_val)
    
    final_na = df.isna().sum().sum()
    logger.info(f"Missing values: {initial_na} → {final_na}")
    return df


def decode_bytes_column(col_data: pd.Series) -> pd.Series:
    """Safely decode byte strings to UTF-8."""
    if col_data.dtype == 'object':
        return col_data.apply(
            lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
        )
    return col_data


def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file."""
    logger.info(f"Loading CSV: {filepath}")
    return pd.read_csv(filepath)


def load_data_file(filepath: str) -> pd.DataFrame:
    """Load .data file with automatic delimiter detection."""
    logger.info(f"Loading .data file: {filepath}")
    separators = [',', ' ', '\t']
    df = None
    
    for sep in separators:
        try:
            df = pd.read_csv(filepath, sep=sep, header=None)
            logger.info(f"Successfully parsed with delimiter: '{sep}'")
            
            expected_cols = list(range(df.shape[1]))
            if list(df.columns) == expected_cols:
                df.columns = [f'feature_{i}' for i in range(df.shape[1])]
                logger.info(f"Auto-generated column names")
            return df
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
    
    raise ValueError(f"Could not parse .data file with separators: {separators}")


def load_arff(filepath: str) -> pd.DataFrame:
    """Load ARFF file with proper byte string handling."""
    logger.info(f"Loading ARFF: {filepath}")
    
    try:
        data, meta = loadarff(filepath)
        df = pd.DataFrame(data)
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = decode_bytes_column(df[col])
        
        logger.info(f"ARFF loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        raise ValueError(f"Failed to load ARFF file: {str(e)}")


def find_target_column(df: pd.DataFrame, candidates: list = None) -> str:
    """Intelligently find the target column."""
    if candidates is None:
        candidates = [
            'target', 'class', 'outcome', 'Class', 'binaryClass', 'status',
            'Target', 'TR', 'speaker', 'Home/Away', 'Outcome', 
            'Leaving_Certificate', 'technology', 'signal', 'label', 'Label',
            'click', 'percent_pell_grant', 'Survival', 'diagnosis'
        ]
    
    for col in candidates:
        if col in df.columns:
            logger.info(f"Found target column: '{col}' (explicit match)")
            return col
    
    last_col = df.columns[-1]
    unique_vals = df[last_col].nunique()
    total_rows = len(df)
    cardinality_ratio = unique_vals / total_rows
    
    if cardinality_ratio < 0.5:
        logger.warning(
            f"Inferring target column: '{last_col}' "
            f"({unique_vals} unique values in {total_rows} rows)"
        )
        return last_col
    
    raise ValueError(
        f"Cannot infer target column. Columns: {list(df.columns)}\n"
        f"Please rename your target to one of: {candidates[:5]}..."
    )


def load_tabular_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load and preprocess tabular data from CSV, .data, or ARFF."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    check_file_size(filepath, max_mb=1000)
    
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext == '.csv':
        df = load_csv(filepath)
    elif file_ext == '.data':
        df = load_data_file(filepath)
    elif file_ext == '.arff':
        df = load_arff(filepath)
    else:
        raise ValueError(
            f"Unsupported file format: '{file_ext}'. Supported: .csv, .data, .arff"
        )
    
    logger.info(f"Loaded data shape: {df.shape}")
    df = smart_fill_missing(df)
    
    target_col = find_target_column(df)
    
    if not np.issubdtype(df[target_col].dtype, np.number):
        logger.info(f"Converting non-numeric target to integers...")
        le = LabelEncoder()
        y = le.fit_transform(df[target_col].astype(str))
        unique_classes = le.classes_.tolist()
    else:
        y = df[target_col].values.astype(int)
        unique_classes = sorted(set(y))
    
    X = df.drop(columns=[target_col]).values
    X = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target distribution: {np.bincount(y)}")
    logger.info(f"Unique classes: {unique_classes}")
    
    return X, y, unique_classes


# ============ VIF CALCULATION ============

def calculate_vif_safe(X_data: np.ndarray) -> np.ndarray:
    """Calculate VIF with proper error handling."""
    df = pd.DataFrame(X_data)
    n_features = df.shape[1]
    vif_values = []
    
    for i in range(n_features):
        try:
            vif = variance_inflation_factor(df.values, i)
            if np.isnan(vif) or np.isinf(vif):
                vif = 1.0
        except:
            vif = 1.0
        vif_values.append(vif)
    
    vif_values = np.array(vif_values)
    vif_values = np.clip(vif_values, 1.0, 100.0)
    return vif_values


# ============ MODEL ARCHITECTURES ============

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
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
    def __init__(self, n_features: int, tab_latent_size: int, num_classes: int):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(n_features, tab_latent_size)
        self.fc2 = nn.Linear(tab_latent_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        tab_latent = self.relu(self.fc1(x))
        x = self.fc2(tab_latent)
        return tab_latent, x


class VIFInitialization(nn.Module):
    def __init__(self, input_dim: int, vif_values: np.ndarray):
        super(VIFInitialization, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, input_dim + 4)
        self.fc2 = nn.Linear(input_dim + 4, input_dim)

        vif_tensor = torch.tensor(vif_values, dtype=torch.float32)
        vif_tensor = vif_tensor / (vif_tensor.mean() + 1e-6)
        inv_vif = 1.0 / torch.clamp(vif_tensor, min=1.0)

        with torch.no_grad():
            for i in range(self.fc1.weight.data.shape[0]):
                self.fc1.weight.data[i, :] = inv_vif[i % len(inv_vif)] / (input_dim + 4)
        logger.info("✅ VIFInitialization: weights set using inverse VIF values.")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class CVAEWithTabEmbedding(nn.Module):
    def __init__(self, n_features: int, tab_latent_size: int, num_classes: int, 
                 vif_values: np.ndarray = None, device: torch.device = None):
        super(CVAEWithTabEmbedding, self).__init__()
        self.device = device
        
        self.mlp = SimpleMLP(n_features, tab_latent_size, num_classes)
        
        if vif_values is not None:
            self.vif_model = VIFInitialization(n_features, vif_values)
        else:
            self.vif_model = None
        
        self.encoder = nn.Sequential(
            nn.Linear(28*28 + tab_latent_size + n_features, 128),
            nn.ReLU(),
            nn.Linear(128, tab_latent_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(tab_latent_size + tab_latent_size + n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
        
        self.final_classifier = SimpleCNN(num_classes=num_classes)

    def encode(self, x, tab_embedding, vif_embedding):
        return self.encoder(torch.cat([x, tab_embedding, vif_embedding], dim=1))
    
    def decode(self, z, tab_embedding, vif_embedding):
        return self.decoder(torch.cat([z, tab_embedding, vif_embedding], dim=1))
    
    def forward(self, x, tab_data):
        if self.vif_model is not None:
            vif_embedding = self.vif_model(tab_data)
        else:
            vif_embedding = tab_data
        
        tab_embedding, tab_pred = self.mlp(tab_data)
        z = self.encode(x, tab_embedding, vif_embedding)
        recon_x = self.decode(z, tab_embedding, vif_embedding)
        img_pred = self.final_classifier(recon_x.view(-1, 1, 28, 28))
        return recon_x, tab_pred, img_pred


# ============ DATASET UTILITIES ============

class SynchronizedDataset(Dataset):
    def __init__(self, tabular_dataset, image_dataset):
        self.tabular_dataset = tabular_dataset
        self.image_dataset = image_dataset
        assert len(self.tabular_dataset) == len(self.image_dataset), \
            "Datasets must have the same length."

    def __len__(self):
        return len(self.tabular_dataset)

    def __getitem__(self, index):
        tab_data, tab_label = self.tabular_dataset[index]
        img_data, img_label = self.image_dataset[index]
        assert tab_label == img_label, \
            f"Label mismatch: tab_label={tab_label}, img_label={img_label}"
        return tab_data, tab_label, img_data, img_label


# ============ TRAINING & TESTING ============

def loss_function(recon_x, x, tab_pred, tab_labels, img_pred, img_labels):
    """Combined loss function."""
    BCE = F.mse_loss(recon_x, x)
    tab_loss = F.cross_entropy(tab_pred, tab_labels)
    img_loss = F.cross_entropy(img_pred, img_labels)
    return BCE + tab_loss + img_loss


def train(model, train_data_loader, optimizer, epoch, device):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    
    for tab_data, tab_label, img_data, img_label in train_data_loader:
        img_data = img_data.view(-1, 28*28).to(device)
        tab_data = tab_data.to(device)
        img_label = img_label.to(device).long()
        tab_label = tab_label.to(device).long()
        
        optimizer.zero_grad()
        
        random_array = np.random.rand(img_data.shape[0], 28*28)
        x_rand = torch.Tensor(random_array).to(device)
        
        recon_x, tab_pred, img_pred = model(x_rand, tab_data)
        loss = loss_function(recon_x, img_data, tab_pred, tab_label, img_pred, img_label)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    logger.info(f"Epoch {epoch}: Train Loss = {train_loss / len(train_data_loader):.4f}")


def test(model, test_data_loader, epoch, best_accuracy, best_auc, best_epoch, 
         device, num_classes, best_model_path='best_model.pth'):
    """Evaluate on test set."""
    model.eval()
    test_loss = 0
    correct_tab_total = 0
    correct_img_total = 0
    total = 0
    
    all_tab_labels = []
    all_tab_preds = []
    all_img_labels = []
    all_img_preds = []

    with torch.no_grad():
        for tab_data, tab_label, img_data, img_label in test_data_loader:
            img_data = img_data.view(-1, 28*28).to(device)
            tab_data = tab_data.to(device)
            img_label = img_label.to(device).long()
            tab_label = tab_label.to(device).long()
            
            random_array = np.random.rand(img_data.shape[0], 28*28)
            x_rand = torch.Tensor(random_array).to(device)
            
            recon_x, tab_pred, img_pred = model(x_rand, tab_data)
            test_loss += loss_function(recon_x, img_data, tab_pred, tab_label, img_pred, img_label).item()
            
            tab_probs = F.softmax(tab_pred, dim=1)
            img_probs = F.softmax(img_pred, dim=1)
            
            all_tab_labels.extend(tab_label.cpu().numpy())
            all_tab_preds.extend(tab_probs.cpu().numpy())
            all_img_labels.extend(img_label.cpu().numpy())
            all_img_preds.extend(img_probs.cpu().numpy())

            tab_predicted = torch.argmax(tab_pred, dim=1)
            img_predicted = torch.argmax(img_pred, dim=1)
            
            correct_tab_total += (tab_predicted == tab_label).sum().item()
            correct_img_total += (img_predicted == img_label).sum().item()
            total += tab_label.size(0)
    
    test_loss /= len(test_data_loader)
    tab_accuracy = 100 * correct_tab_total / total
    img_accuracy = 100 * correct_img_total / total
    
    # Calculate AUC
    all_tab_preds_arr = np.array(all_tab_preds)
    all_img_preds_arr = np.array(all_img_preds)
    all_tab_labels_arr = np.array(all_tab_labels)
    all_img_labels_arr = np.array(all_img_labels)

    try:
        if num_classes == 2:
            tab_auc = roc_auc_score(all_tab_labels_arr, all_tab_preds_arr[:, 1])
            img_auc = roc_auc_score(all_img_labels_arr, all_img_preds_arr[:, 1])
        else:
            tab_auc = roc_auc_score(all_tab_labels_arr, all_tab_preds_arr, 
                                   multi_class="ovr", average="macro")
            img_auc = roc_auc_score(all_img_labels_arr, all_img_preds_arr, 
                                   multi_class="ovr", average="macro")
    except Exception as e:
        logger.warning(f"AUC calculation failed: {e}")
        tab_auc = img_auc = 0.0

    logger.info(
        f"Epoch {epoch}: Test Loss = {test_loss:.4f} | "
        f"Tab Acc = {tab_accuracy:.2f}% | Img Acc = {img_accuracy:.2f}% | "
        f"Tab AUC = {tab_auc:.4f} | Img AUC = {img_auc:.4f}"
    )

    # Save best model
    if img_accuracy > best_accuracy:
        best_accuracy = img_accuracy
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"✅ Best model saved with accuracy {best_accuracy:.2f}%")
    
    if img_auc > best_auc:
        best_auc = img_auc

    return best_accuracy, best_auc, best_epoch


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description="Table2Image - Tabular Data Classification")
    parser.add_argument('--input_file', type=str, required=True, 
                       help='Path to dataset (.csv, .data, or .arff)')
    parser.add_argument('--save_dir', type=str, required=True, 
                       help='Path to save the final model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--dataset_root', type=str, default='/tmp/datasets', 
                       help='Root directory for MNIST/FashionMNIST')
    args = parser.parse_args()

    # Parameters
    EPOCH = args.epochs
    BATCH_SIZE = args.batch_size
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DATASET_ROOT = args.dataset_root
    
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Epochs: {EPOCH}, Batch Size: {BATCH_SIZE}")

    # Load data
    logger.info("=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)
    X, y, unique_classes = load_tabular_data(args.input_file)
    
    num_classes = len(unique_classes)
    n_cont_features = X.shape[1]
    tab_latent_size = n_cont_features + 4
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Number of features: {n_cont_features}")
    logger.info(f"Tab latent size: {tab_latent_size}")

    # Calculate VIF
    logger.info("Calculating VIF values...")
    X_sample = X[:min(1000, len(X))]
    vif_values = calculate_vif_safe(X_sample)
    logger.info(f"VIF values: {vif_values[:5]}...")

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create tabular datasets
    train_tabular_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_tabular_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    # Load image datasets
    logger.info("Loading MNIST/FashionMNIST datasets...")
    fashionmnist_dataset = datasets.FashionMNIST(
        root=DATASET_ROOT, train=True, download=True,
        transform=transforms.ToTensor()
    )
    
    mnist_dataset = datasets.MNIST(
        root=DATASET_ROOT, train=True, download=True,
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

    # Prepare image datasets
    valid_labels = {i for i in range(num_classes)}
    filtered_fashion = Subset(fashionmnist_dataset, 
                             [i for i, (_, label) in enumerate(fashionmnist_dataset) 
                              if label in valid_labels])
    filtered_mnist = Subset(modified_mnist_dataset,
                           [i for i, (_, label) in enumerate(modified_mnist_dataset) 
                            if label in valid_labels])

    combined_dataset = ConcatDataset([filtered_fashion, filtered_mnist])

    # Align indices
    train_tabular_label_counts = torch.bincount(
        train_tabular_dataset.tensors[1], minlength=num_classes
    )
    test_tabular_label_counts = torch.bincount(
        test_tabular_dataset.tensors[1], minlength=num_classes
    )

    indices_by_label = {label: [] for label in range(num_classes)}
    for i, (_, label) in enumerate(combined_dataset):
        if label not in indices_by_label:
            indices_by_label[label] = []
        indices_by_label[label].append(i)

    repeated_indices = {
        label: list(itertools.islice(
            itertools.cycle(indices_by_label[label]),
            train_tabular_label_counts[label] + test_tabular_label_counts[label]
        ))
        for label in indices_by_label
    }

    aligned_train_indices = []
    aligned_test_indices = []

    for label in valid_labels:
        train_tab_indices = [i for i, lbl in enumerate(y_train) if lbl == label]
        test_tab_indices = [i for i, lbl in enumerate(y_test) if lbl == label]

        train_img_indices = repeated_indices[label][:train_tabular_label_counts[label]]
        test_img_indices = repeated_indices[label][
            train_tabular_label_counts[label]:
            train_tabular_label_counts[label] + test_tabular_label_counts[label]
        ]

        aligned_train_indices.extend(list(zip(train_tab_indices, train_img_indices)))
        aligned_test_indices.extend(list(zip(test_tab_indices, test_img_indices)))

    train_filtered_tab = Subset(train_tabular_dataset, 
                               [idx[0] for idx in aligned_train_indices])
    train_filtered_img = Subset(combined_dataset, 
                               [idx[1] for idx in aligned_train_indices])
    test_filtered_tab = Subset(test_tabular_dataset, 
                              [idx[0] for idx in aligned_test_indices])
    test_filtered_img = Subset(combined_dataset, 
                              [idx[1] for idx in aligned_test_indices])

    train_sync_dataset = SynchronizedDataset(train_filtered_tab, train_filtered_img)
    test_sync_dataset = SynchronizedDataset(test_filtered_tab, test_filtered_img)

    train_loader = DataLoader(train_sync_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_sync_dataset, batch_size=BATCH_SIZE)

    logger.info(f"Train samples: {len(train_sync_dataset)}")
    logger.info(f"Test samples: {len(test_sync_dataset)}")

    # Create model
    logger.info("=" * 60)
    logger.info("CREATING MODEL")
    logger.info("=" * 60)
    
    model = CVAEWithTabEmbedding(
        n_features=n_cont_features,
        tab_latent_size=tab_latent_size,
        num_classes=num_classes,
        vif_values=vif_values,
        device=DEVICE
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Train
    logger.info("=" * 60)
    logger.info("TRAINING")
    logger.info("=" * 60)
    
    best_accuracy = 0
    best_auc = 0
    best_epoch = 0
    saving_path = args.save_dir + '.pt'

    for epoch in range(1, EPOCH + 1):
        train(model, train_loader, optimizer, epoch, DEVICE)
        best_accuracy, best_auc, best_epoch = test(
            model, test_loader, epoch, best_accuracy, best_auc, best_epoch,
            DEVICE, num_classes, best_model_path=saving_path
        )

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best model accuracy: {best_accuracy:.4f} at epoch {best_epoch}")
    logger.info(f"Best AUC: {best_auc:.4f}")
    logger.info(f"Model saved to: {saving_path}")


if __name__ == "__main__":
    main()
