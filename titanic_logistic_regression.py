import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ================== 1. 数据加载与预处理 ==================
df = pd.read_csv("data/train.csv")

# 查看数据基本信息（可选）
print("数据集形状:", df.shape)
print(df.head())

# --- 特征工程 ---
# 1. 填充缺失值
df['Age'].fillna(df['Age'].median(), inplace=True)          # 年龄用中位数填充
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True) # 登船港口用众数填充
df['Fare'].fillna(df['Fare'].median(), inplace=True)         # 票价用中位数填充
# Cabin 缺失太多，删除该列
df.drop('Cabin', axis=1, inplace=True)

# 2. 名字特征：提取称谓（Title）
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# 合并罕见称谓
title_mapping = {
    'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
    'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
    'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
    'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
    'Capt': 'Rare', 'Sir': 'Rare'
}
df['Title'] = df['Title'].map(title_mapping).fillna('Rare')

# 3. 家庭规模
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# 4. 是否独自一人
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# 5. 删除不需要的列
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# 6. 将分类变量转变为数值（独热编码）
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)

# 7. 分离特征与标签
X = df.drop('Survived', axis=1)
y = df['Survived']

print("\n预处理后特征形状:", X.shape)
print("特征列:", X.columns.tolist())

# ================== 2. 划分训练集和测试集 ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================== 3. 特征标准化 ==================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================== 4. 转换为 PyTorch 张量 ==================
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ================== 5. 定义逻辑回归模型 ==================
# 使用 nn.BCEWithLogitsLoss 时，模型不需要加 Sigmoid，因为 loss 内部包含 sigmoid
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)   # 输出一个 logit

    def forward(self, x):
        return self.linear(x)    # 原始输出，不经过 sigmoid

input_dim = X_train_tensor.shape[1]
model = LogisticRegression(input_dim)
print("\n模型结构:\n", model)

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()   # 包含 sigmoid + 二元交叉熵
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ================== 6. 训练模型 ==================
epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # 评估模式
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor).item()
        test_losses.append(test_loss)

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Test Loss: {test_loss:.4f}')

# ================== 7. 绘制损失曲线 ==================
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ================== 8. 模型评估 ==================
model.eval()
with torch.no_grad():
    # 训练集预测
    train_logits = model(X_train_tensor)
    train_probs = torch.sigmoid(train_logits)           # 转换为概率
    train_preds = (train_probs >= 0.5).int().numpy()   # 阈值0.5

    # 测试集预测
    test_logits = model(X_test_tensor)
    test_probs = torch.sigmoid(test_logits)
    test_preds = (test_probs >= 0.5).int().numpy()

# 准确率
train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)
print("\n==================== 准确率 ====================")
print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")

# 混淆矩阵与分类报告
print("\n测试集混淆矩阵:")
print(confusion_matrix(y_test, test_preds))
print("\n分类报告:")
print(classification_report(y_test, test_preds, target_names=['Not Survived', 'Survived']))

# ================== 9. 保存模型 ==================
torch.save(model.state_dict(), 'titanic_logistic_regression.pth')
print("\n模型已保存为 titanic_logistic_regression.pth")