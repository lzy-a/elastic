import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from lstm import LSTMModel
from torch.utils.data import Dataset, DataLoader


def generate_sequences(df: pd.DataFrame, tw: int, pw: int, target_columns, drop_targets=False):
    '''
   df: Pandas DataFrame of the univariate time-series
   tw: Training Window - Integer defining how many steps to look back
   pw: Prediction Window - Integer defining how many steps forward to predict

   returns: dictionary of sequences and targets for all sequences
   '''
    data = dict()  # Store results into a dictionary
    L = len(df)
    for i in range(L - tw):
        # Option to drop target from dataframe
        if drop_targets:
            df.drop(target_columns, axis=1, inplace=True)

        # Get current sequence
        sequence = df[i:i + tw].values
        # Get values right after the current sequence
        target = df[i + tw:i + tw + pw][target_columns].values
        data[i] = {'sequence': sequence, 'target': target}
    return data


class SequenceDataset(Dataset):

    def __init__(self, df):
        self.data = df

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.Tensor(sample['sequence']), torch.Tensor(sample['target'])

    def __len__(self):
        return len(self.data)


date_range = pd.date_range(start='1/1/2023', end='12/31/2023')
traffic_data = pd.read_csv('history_data.csv')
data = generate_sequences(traffic_data, tw=24, pw=24, target_columns=[1])

dataset = SequenceDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

input_size = data.shape[1]
hidden_size = 100  # LSTM隐藏层的大小
output_size = 1  # 输出特征的维度（这里假设为1）
num_epochs = 100  # 训练的轮数
learning_rate = 0.001  # 学习率

model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

# 训练模型
for epoch in range(num_epochs):
    for batch in dataloader:
        print(batch)
        # 前向传播
        outputs = model(batch)
        loss = criterion(outputs, torch.tensor([1] * len(batch)))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
