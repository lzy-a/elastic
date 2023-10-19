import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from lstm import LSTMModel

# 定义全天的24小时
hours = np.arange(24)

# 设置一年日期范围
date_range = pd.date_range(start='1/1/2023', end='12/31/2023')

# 初始化一个空的DataFrame用于存储流量数据
traffic_data = pd.DataFrame(index=date_range)

# 为每天的1-7点设置流量为1000，其他时间段流量为6000
for hour in range(24):
    if hour < 7:
        traffic_data.loc[date_range, hour] = 1000
    else:
        traffic_data.loc[date_range, hour] = 6000

    # 对全时段应用10%的随机波动
for hour in range(24):
    traffic_data.loc[date_range, hour] = traffic_data.loc[date_range, hour] * (1 + np.random.rand(len(date_range)) / 10)
    traffic_data.loc[date_range, hour] = traffic_data.loc[date_range, hour].round(2)  # 四舍五入保留两位小数


traffic_data.to_csv('history_data.csv', index=False)

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
