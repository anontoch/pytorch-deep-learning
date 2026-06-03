import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("data/advertising.csv")
x_raw = data['wechat'].values.astype(np.float32).reshape(-1,1) #[200] ->[200,1]
y_raw = data['sales'].values.astype(np.float32).reshape(-1,1) #[200] ->[200,1]
#数据标准化
x_mean = x_raw.mean()
x_std = x_raw.std()
x_norm = (x_raw - x_mean)/x_std
#从numpy转tensor
x = torch.from_numpy(x_norm)
y = torch.from_numpy(y_raw)
print(x.shape,y.shape)
#定义线性回归模型
model = nn.Linear(in_features=1,out_features=1)
#定义损失函数
criterion = nn.MSELoss()
#定义优化器
optimizer = optim.SGD(model.parameters(),lr=0.01)

#开始训练模型
num_epochs = 1000
loss_history = []

for epoch in range(num_epochs):
    #前向传播，计算预测结果
    y_pred = model(x)
    #计算损失
    loss = criterion(y_pred,y)
    # 反向传播：清零梯度、计算梯度、更新参数
    optimizer.zero_grad()
    loss.backward() #对损失进行反向传播
    optimizer.step() #更新参数

    loss_history.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

w_norm = model.weight.item()
b_norm = model.bias.item()
w_raw = w_norm / x_std
b_raw = b_norm - w_norm * x_mean / x_std
print(f'原始尺度方程: sales = {w_raw:.4f} * wechat + {b_raw:.4f}')

plt.rcParams['font.sans-serif'] = ['SimHei']        # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False          # 正常显示负号
#可视化
plt.scatter(x_raw,y,color='blue',alpha=0.6,label='原始数据')
# 生成一条连续的 X 值（原始尺度）
x_line_raw = np.linspace(x_raw.min(),x_raw.max(),100).reshape(-1,1)
# 标准化后送入模型预测
x_line_norm = (x_line_raw - x_mean)/ x_std
x_line_tensor = torch.from_numpy(x_line_norm)
#模型预测
y_line = model(x_line_tensor).detach().numpy()
plt.plot(x_line_raw, y_line, color='red', linewidth=2, label='拟合直线')
plt.xlabel('微信广告投入')
plt.ylabel('销售额')
plt.title('一元线性回归：微信广告投入 vs 销售额')
plt.legend()
plt.grid(True)
plt.show()

#绘制损失曲线图
plt.figure()
plt.plot(range(num_epochs),loss_history)
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.title('训练过程中损失函数变化')
plt.grid(True)
plt.show()