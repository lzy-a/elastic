import os

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from lstm import LSTMModel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


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


# 画出预测结果
def plot_predict():
    # 对原来对数据集进行推理
    model.eval()
    preds = []
    labels = []
    df = pd.read_csv('data_hour.csv')
    df = df.tail(100)
    df_normalized = scaler.fit_transform(df)
    df = pd.DataFrame(df_normalized, columns=df.columns)
    test_data = generate_sequences(df, tw=input_size, pw=output_size, target_columns="0")
    # 取test_data的最后100个数据
    dataset = SequenceDataset(test_data)
    dataloader = DataLoader(dataset, batch_size=1)
    print(len(dataloader))
    for x, y in dataloader:
        with torch.no_grad():
            x, y = x.to(device), y.squeeze().to(device)
            y_hat = model(x).squeeze()
            # 打印y_hat的值
            print(x, y, y_hat)
            # 把y_hat 转移到cpu, 拼接到preds中
            # 把非0维的结果进行拼接

            preds.append(y_hat.cpu().numpy())
            labels.append(y.cpu().numpy())

    # 将列表转换为numpy数组
    # preds = np.concatenate(preds, axis=0)
    # labels = np.concatenate(labels, axis=0)
    preds = scaler.inverse_transform(preds)
    labels = scaler.inverse_transform(labels)
    print(preds)
    print(labels)
    # plot the results
    plt.plot(preds, label='predictions')
    plt.plot(labels, label='actual')
    plt.legend()
    plt.show()
    plt.savefig('lstm.png')


if __name__ == '__main__':
    split = 0.8
    BATCH_SIZE = 256
    input_size = 24
    hidden_size = 100  # LSTM隐藏层的大小
    output_size = 1  # 输出特征的维度（这里假设为1）
    n_epochs = 10  # 训练的轮数
    learning_rate = 0.001  # 学习率
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 判断是否有GPU加速

    traffic_data = pd.read_csv('data_hour.csv')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    traffic_data_normalized = scaler.fit_transform(traffic_data,columns=traffic_data.columns)
    traffic_data = pd.DataFrame(traffic_data_normalized)
    data = generate_sequences(traffic_data, tw=input_size, pw=output_size, target_columns="0")
    print("data gerneated")
    dataset = SequenceDataset(data)
    dataloader = DataLoader(dataset, batch_size=32)
    train_len = int(len(dataset) * split)
    lens = [train_len, len(dataset) - train_len]
    train_ds, test_ds = random_split(dataset, lens)
    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = LSTMModel(1, n_hidden=hidden_size, n_outputs=output_size, sequence_len=input_size, n_lstm_layers=5,
                      device=device).to(device)
    if os.path.exists('lstm.pt'):
        model.load_state_dict(torch.load('lstm.pt', map_location="cpu"))
    criterion = nn.MSELoss()  # 均方误差损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

    # 训练模型
    # Lists to store training and validation losses
    t_losses, v_losses = [], []
    # Loop over epochs
    for epoch in range(n_epochs):
        train_loss, valid_loss = 0.0, 0.0

        # train step
        model.train()
        # Loop over train dataset
        for x, y in trainloader:
            optimizer.zero_grad()
            # move inputs to device
            x = x.to(device)
            y = y.squeeze().to(device)
            # Forward Pass
            preds = model(x).squeeze()
            loss = criterion(preds, y)  # compute batch loss
            print(f'loss: {loss.item()}')
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_loss = train_loss / len(trainloader)
        t_losses.append(epoch_loss)
        torch.save(model.state_dict(), 'lstm.pt')
        # validation step
        model.eval()
        # Loop over validation dataset
        for x, y in testloader:
            with torch.no_grad():
                x, y = x.to(device), y.squeeze().to(device)
                preds = model(x).squeeze()
                error = criterion(preds, y)
            valid_loss += error.item()

        valid_loss = valid_loss / len(testloader)
        v_losses.append(valid_loss)

        if epoch % 10 == 1:
            plot_predict()

        print(
            f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')
