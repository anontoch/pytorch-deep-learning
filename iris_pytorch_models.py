import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from torch.utils.data import TensorDataset, DataLoader

# =====================================================
# 1. Random Seed
# =====================================================

torch.manual_seed(42)
np.random.seed(42)

# =====================================================
# 2. Load Dataset
# =====================================================

iris = load_iris()

X = iris.data
y = iris.target

class_names = iris.target_names

print("Dataset Shape:", X.shape)
print("Classes:", class_names)

# =====================================================
# 3. Train/Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# 4. Feature Scaling
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# 5. Convert to Tensor
# =====================================================

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# =====================================================
# 6. Create DataLoader
# =====================================================

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

# =====================================================
# 7. Logistic Regression Model
# =====================================================

class LogisticRegressionModel(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):

        # Returns logits
        return self.linear(x)

# =====================================================
# 8. Neural Network Model
# =====================================================

class NeuralNetworkModel(nn.Module):

    def __init__(self, input_dim, hidden1=16, hidden2=8, num_classes=3):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, hidden1),
            nn.ReLU(),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),

            nn.Linear(hidden2, num_classes)

        )

    def forward(self, x):

        return self.network(x)

# =====================================================
# 9. Training Function
# =====================================================

def train_model(model, train_loader, test_loader, epochs=100, lr=0.01):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    train_losses = []
    test_losses = []

    for epoch in range(epochs):

        # -----------------------------
        # Training
        # -----------------------------

        model.train()

        running_loss = 0.0

        for inputs, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        train_losses.append(train_loss)

        # -----------------------------
        # Evaluation
        # -----------------------------

        model.eval()

        total_loss = 0.0

        with torch.no_grad():

            for inputs, labels in test_loader:

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)

        test_loss = total_loss / len(test_loader.dataset)

        test_losses.append(test_loss)

        # -----------------------------
        # Print Progress
        # -----------------------------

        if (epoch + 1) % 20 == 0:

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Test Loss: {test_loss:.4f}"
            )

    # =================================================
    # Accuracy Calculation
    # =================================================

    model.eval()

    with torch.no_grad():

        train_predictions = torch.argmax(
            model(X_train_tensor),
            dim=1
        ).numpy()

        test_predictions = torch.argmax(
            model(X_test_tensor),
            dim=1
        ).numpy()

    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    return (
        train_accuracy,
        test_accuracy,
        test_predictions,
        train_losses,
        test_losses
    )

# =====================================================
# 10. Initialize Models
# =====================================================

input_dim = X_train.shape[1]
num_classes = len(np.unique(y))

# =====================================================
# 11. Train Logistic Regression
# =====================================================

print("\n========== Logistic Regression ==========\n")

logistic_model = LogisticRegressionModel(
    input_dim,
    num_classes
)

(
    logreg_train_acc,
    logreg_test_acc,
    logreg_predictions,
    logreg_train_losses,
    logreg_test_losses

) = train_model(
    logistic_model,
    train_loader,
    test_loader,
    epochs=100,
    lr=0.1
)

print("\nLogistic Regression Results")

print(f"Train Accuracy: {logreg_train_acc:.4f}")
print(f"Test Accuracy : {logreg_test_acc:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        logreg_predictions,
        target_names=class_names
    )
)

# =====================================================
# 12. Train Neural Network
# =====================================================

print("\n========== Neural Network ==========\n")

neural_model = NeuralNetworkModel(
    input_dim=input_dim,
    hidden1=16,
    hidden2=8,
    num_classes=num_classes
)

(
    nn_train_acc,
    nn_test_acc,
    nn_predictions,
    nn_train_losses,
    nn_test_losses

) = train_model(
    neural_model,
    train_loader,
    test_loader,
    epochs=100,
    lr=0.01
)

print("\nNeural Network Results")

print(f"Train Accuracy: {nn_train_acc:.4f}")
print(f"Test Accuracy : {nn_test_acc:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        nn_predictions,
        target_names=class_names
    )
)

# =====================================================
# 13. Plot Loss Curves
# =====================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Logistic Regression Loss
plt.subplot(1, 2, 1)

plt.plot(logreg_train_losses, label="Train Loss")
plt.plot(logreg_test_losses, label="Test Loss")

plt.title("Logistic Regression Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid()

# Neural Network Loss
plt.subplot(1, 2, 2)

plt.plot(nn_train_losses, label="Train Loss")
plt.plot(nn_test_losses, label="Test Loss")

plt.title("Neural Network Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# =====================================================
# 14. Save Models
# =====================================================

torch.save(
    logistic_model.state_dict(),
    "iris_logistic_model.pth"
)

torch.save(
    neural_model.state_dict(),
    "iris_neural_network.pth"
)

print("\nModels Saved Successfully!")