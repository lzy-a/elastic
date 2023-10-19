import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from lstm import LSTMModel

date_range = pd.date_range(start='1/1/2023', end='12/31/2023')
traffic_data = pd.read_csv('history_data.csv', index=date_range, columns=range(24))
data = torch.tensor(traffic_data)
# 将数据划分为输入和目标变量
input_data = data[:, :-1].float()
target_data = data[:, 1:].float()

# 将输入数据和目标数据转换为适合LSTM的格式
input_data = input_data.reshape(-1, 1, 24)
target_data = target_data.reshape(-1, 1)

input_size = 1  # 输入特征的维度（这里假设为1）
hidden_size = 100  # LSTM隐藏层的大小
output_size = 1  # 输出特征的维度（这里假设为1）
num_epochs = 100  # 训练的轮数
learning_rate = 0.001  # 学习率

model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

# 训练模型
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(input_data)
    loss = criterion(outputs, target_data)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')