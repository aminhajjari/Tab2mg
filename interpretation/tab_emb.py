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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import os
import argparse

###########################변경 사항#############################
num_classes = 2  # 클래스 개수 수정

class TABMLP(nn.Module):
    def __init__(self, n_cont_features=8, tab_latent_size=12, num_classes=5):
        super(TABMLP, self).__init__()
        self.fc1 = nn.Linear(n_cont_features, tab_latent_size)  # 8 features as input
        self.fc2 = nn.Linear(tab_latent_size, num_classes)  # Output layer for multi-class classification
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
###########################변경 사항#############################

def main():
    # argparse로 외부 인자 받기
    parser = argparse.ArgumentParser(description="Train a diabetes model with a CSV file.")
    parser.add_argument('--csv', required=True, help="Path to the input CSV file.")
    parser.add_argument('--csv_name', type=str, required=True, help='Path to the dataset (csv)')
    args = parser.parse_args()

    csv_file = args.csv

    # CSV 파일 로드
    csv_name = args.csv_name

    df = pd.read_csv(csv_file)
    target_col_candidates = ['target', 'class', 'outcome', 'Class', 'binaryClass', 'status', 'Target', 'TR', 'speaker', 'Home/Away', 'Outcome', 'Leaving_Certificate', 'technology', 'signal', 'label', 'Label', 'click', 'percent_pell_grant', 'Survival']
    target_col = next((col for col in df.columns if col.lower() in target_col_candidates), None)
    if target_col == None:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    else:
        y = df.loc[:, target_col].values
        X = df.drop(target_col, axis=1).values

    # Train/Test 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 데이터 표준화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # PyTorch 텐서로 변환
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # CrossEntropyLoss requires `long`
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 모델 초기화
    n_cont_features = X.shape[1]
    tab_latent_size = n_cont_features + 4
    model = TABMLP(n_cont_features=n_cont_features, tab_latent_size=tab_latent_size, num_classes=num_classes)

    # Loss와 Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training the model
    best_acc = 0.0
    num_epochs = 50

    best_model_path = f"{csv_name}/{csv_name}_tab.pt"

    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred_train = torch.argmax(model(X_train_tensor), dim=1)
            y_pred_test = torch.argmax(model(X_test_tensor), dim=1)

            train_acc = accuracy_score(y_train_tensor.numpy(), y_pred_train.numpy())
            test_acc = accuracy_score(y_test_tensor.numpy(), y_pred_test.numpy())
            #test_auc = roc_auc_score(
#                 pd.get_dummies(y_test_tensor.numpy()), 
#                 model(X_test_tensor).softmax(dim=1).numpy(), 
#                 multi_class="ovr"
#             )

            if test_acc > best_acc:
                best_acc = test_acc
                # Best 모델 저장
                torch.save(model, best_model_path)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}, "
                  f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # Best 모델 로드 및 평가
#     best_model = TABMLP(tab_latent_size=tab_latent_size, num_classes=num_classes)
    best_model = torch.load(best_model_path)
    best_model.eval()

    with torch.no_grad():
        y_pred_test = torch.argmax(best_model(X_test_tensor), dim=1)
        best_acc = accuracy_score(y_test_tensor.numpy(), y_pred_test.numpy())
#         best_auc = roc_auc_score(
#             pd.get_dummies(y_test_tensor.numpy()), 
#             best_model(X_test_tensor).softmax(dim=1).numpy(), 
#             multi_class="ovr"
#         )

    print(f"Best Test Accuracy: {best_acc:.4f}")
    print(f"Best model saved to: {best_model_path}")

if __name__ == "__main__":
    main()

# -


