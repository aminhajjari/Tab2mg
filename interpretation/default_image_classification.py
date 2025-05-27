import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST, MNIST
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.metrics import roc_auc_score, accuracy_score
import os

# Argument parser
parser = argparse.ArgumentParser(description="Welcome to DualSHAP")
parser.add_argument('--num_classes', type=int, required=True, help='# of classes')
args = parser.parse_args()

num_classes = args.num_classes

class SimpleCNN_seq(nn.Module):
    def __init__(self, num_classes=15):
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
            nn.Linear(128, num_classes)  # Updated output to match total classes (5 from FashionMNIST + 10 from MNIST)
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Pass through conv layers
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)  # Pass through fully connected layers
        return x


# Dataset preprocessing and DataLoader setup
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms

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

# Custom dataset that offsets MNIST labels by 10
class ModifiedLabelDataset(Dataset):
    def __init__(self, dataset, label_offset=10):
        self.dataset = dataset
        self.label_offset = label_offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label + self.label_offset

# Create modified MNIST dataset with labels offset by 10
modified_mnist_dataset = ModifiedLabelDataset(mnist_dataset, label_offset=10)

valid_labels = {i for i in range(num_classes)}

# Filter FashionMNIST dataset
filtered_fashion = Subset(fashionmnist_dataset, 
                          [i for i, (_, label) in enumerate(fashionmnist_dataset) if label in valid_labels])

# Filter MNIST dataset with offset labels
filtered_mnist = Subset(modified_mnist_dataset, 
                        [i for i, (_, label) in enumerate(modified_mnist_dataset) if label in valid_labels])

# Combine FashionMNIST and MNIST
combined_dataset = ConcatDataset([filtered_fashion, filtered_mnist])

# DataLoader setup
train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
test_loader = train_loader

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN_seq(num_classes=15).to(device)  # Set num_classes to 15 for combined labels

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
best_acc = 0.0  # Variable to store the best accuracy
save_path = 'image_classification_' + str(num_classes) + '.pt'  # Path to save the best model


# Training function
def train(model, train_loader, test_loader, criterion, optimizer, device, epochs=5):
    global best_acc
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass and loss calculation
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
        # Evaluate model on test data
        acc = evaluate(model, test_loader, device)
        print(f'Epoch [{epoch+1}/{epochs}], Accuracy: {acc:.2f}%')
        
        # Save model if accuracy improves
        if acc > best_acc:
            best_acc = acc
            torch.save(model, save_path)
            print(f"Best model saved with accuracy: {best_acc:.2f}%")


# Evaluation function (accuracy, auc calculation)
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.softmax(dim=1).cpu().numpy()[:, 1])  # Using softmax for probabilities
    
    # Calculate accuracy and AUC
    acc = 100 * correct / total
#     auc = roc_auc_score(all_labels, all_outputs, multi_class='ovr')
    
    return acc


# Start training and evaluating
train(model, train_loader, test_loader, criterion, optimizer, device, epochs=5)

# Load the best model
print(f"Loading best model with accuracy: {best_acc:.2f}%")
model = torch.load(save_path)
acc = evaluate(model, test_loader, device)
print(acc)




