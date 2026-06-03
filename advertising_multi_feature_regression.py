import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv("data/advertising.csv")
x_raw = data[['wechat','weibo','others']].values.astype(np.float32)
y_raw = data['sales'].values.astype(np.float32).reshape(-1,1)

# 划分训练/测试集
x_train_raw, x_test_raw, y_train, y_test = train_test_split(
    x_raw, y_raw, test_size=0.2, random_state=42
)

# 特征标准化（基于训练集）
x_mean = x_train_raw.mean(axis=0)
x_std = x_train_raw.std(axis=0)
x_std[x_std == 0] = 1    # 防止除零
x_train = (x_train_raw - x_mean) / x_std
x_test = (x_test_raw - x_mean) / x_std

# 转为张量
x_train_t = torch.from_numpy(x_train)
y_train_t = torch.from_numpy(y_train)
x_test_t = torch.from_numpy(x_test)
y_test_t = torch.from_numpy(y_test)

model = nn.Linear(in_features=3, out_features=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练
num_epochs = 2000
train_losses, test_losses = [], []

for epoch in range(num_epochs):
    model.train()
    pred_train = model(x_train_t)
    loss_train = criterion(pred_train, y_train_t)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    train_losses.append(loss_train.item())

    model.eval()
    with torch.no_grad():
        pred_test = model(x_test_t)
        loss_test = criterion(pred_test, y_test_t)
        test_losses.append(loss_test.item())

    if (epoch + 1) % 400 == 0:
        print(f'Epoch {epoch + 1:4d} | Train Loss: {loss_train.item():.6f} | Test Loss: {loss_test.item():.6f}')


# 还原原始尺度方程
w_norm = model.weight.detach().numpy().flatten()
b_norm = model.bias.detach().item()
w_raw = w_norm / x_std
b_raw = b_norm - np.sum(w_norm * x_mean / x_std)
print("\n=== 多元线性回归方程（原始尺度） ===")
print(f"sales = {w_raw[0]:.4f} * wechat + {w_raw[1]:.4f} * weibo + {w_raw[2]:.4f} * others + {b_raw:.4f}")

# 测试集预测与评估
model.eval()
with torch.no_grad():
    y_pred_test = model(x_test_t).numpy()
Y_test_np = y_test_t.numpy()
mse = np.mean((y_pred_test - Y_test_np)**2)
r2 = r2_score(Y_test_np, y_pred_test)
print(f"\n测试集 MSE: {mse:.4f}, R²: {r2:.4f}")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses,label='训练损失')
plt.plot(test_losses,label='测试损失')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('损失曲线')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(Y_test_np,y_pred_test,alpha=0.6)
plt.plot([Y_test_np.min(),Y_test_np.max()],[Y_test_np.min(),Y_test_np.max()],'r--')
plt.xlabel('真实销售额')
plt.ylabel('预测销售额')
plt.title('预测vs真实')
plt.grid(True)

plt.tight_layout()
plt.show()