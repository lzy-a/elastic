import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from lstm import LSTMModel
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


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


if __name__ == '__main__':
    split = 0.8
    BATCH_SIZE = 32
    input_size = 180
    hidden_size = 100  # LSTM隐藏层的大小
    output_size = 1  # 输出特征的维度（这里假设为1）
    n_epochs = 100  # 训练的轮数
    learning_rate = 0.001  # 学习率
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 判断是否有GPU加速

    traffic_data = pd.read_csv('history_data.csv')
    data = generate_sequences(traffic_data, tw=input_size, pw=output_size, target_columns="0")
    # print(data)
    dataset = SequenceDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    train_len = int(len(dataset) * split)
    lens = [train_len, len(dataset) - train_len]
    train_ds, test_ds = random_split(dataset, lens)
    trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


    model = LSTMModel(input_size, hidden_size, output_size).to(device)
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
            print(x.shape)
            print(y.shape)
            optimizer.zero_grad()
            # move inputs to device
            x = x.to(device)
            y = y.squeeze().to(device)
            # Forward Pass
            preds = model(x).squeeze()
            loss = criterion(preds, y)  # compute batch loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = train_loss / len(trainloader)
        t_losses.append(epoch_loss)

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

        print(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')
