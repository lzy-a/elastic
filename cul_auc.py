import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import sys
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from deepfm import deepfm
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def get_auc(loader, model):
    pred, target = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            y_hat = model(x).to(device)
            pred += list(y_hat.cpu().numpy())  # 将y_hat移动到CPU上
            target += list(y.cpu().numpy())  # 将y移动到CPU上
    auc = roc_auc_score(target, pred)
    return auc


def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    # kafka_warmup()
    batch_size = 1024
    lr = 0.0005
    wd = 0.0001

    seed = 1024
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)
    random.seed(seed)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    col_names = ['label'] + dense_features + sparse_features
    df = pd.read_csv('data/small.txt', names=col_names, sep='\t')
    # df = pd.read_csv('data/test.txt', names=col_names, sep='\t')
    feature_names = dense_features + sparse_features

    df[sparse_features] = df[sparse_features].fillna('-1', )
    df[dense_features] = df[dense_features].fillna(0, )
    target = ['label']

    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    df[dense_features] = mms.fit_transform(df[dense_features])

    feat_size1 = {feat: 1 for feat in dense_features}
    feat_size2 = {feat: len(df[feat].unique()) for feat in sparse_features}
    feat_sizes = {}
    feat_sizes.update(feat_size1)
    feat_sizes.update(feat_size2)
    # feat_sizes = {'I1': 1, 'I2': 1, 'I3': 1, 'I4': 1, 'I5': 1, 'I6': 1, 'I7': 1, 'I8': 1, 'I9': 1, 'I10': 1, 'I11': 1,
    #               'I12': 1, 'I13': 1, 'C1': 541, 'C2': 497, 'C3': 43870, 'C4': 25184, 'C5': 145, 'C6': 12, 'C7': 7623,
    #               'C8': 257, 'C9': 3, 'C10': 10997, 'C11': 3799, 'C12': 41312, 'C13': 2796, 'C14': 26, 'C15': 5238,
    #               'C16': 34617, 'C17': 10, 'C18': 2548, 'C19': 1303, 'C20': 4, 'C21': 38618, 'C22': 11, 'C23': 14,
    #               'C24': 12335, 'C25': 51, 'C26': 9527}
    feat_sizes = {'I1': 1, 'I2': 1, 'I3': 1, 'I4': 1, 'I5': 1, 'I6': 1, 'I7': 1, 'I8': 1, 'I9': 1, 'I10': 1, 'I11': 1,
                  'I12': 1, 'I13': 1, 'C1': 1460, 'C2': 583, 'C3': 10131227, 'C4': 2202608, 'C5': 305, 'C6': 24,
                  'C7': 12517, 'C8': 633, 'C9': 3, 'C10': 93145, 'C11': 5683, 'C12': 8351593, 'C13': 3194, 'C14': 27,
                  'C15': 14992, 'C16': 5461306, 'C17': 10, 'C18': 5652, 'C19': 2173, 'C20': 4, 'C21': 7046547,
                  'C22': 18, 'C23': 15, 'C24': 286181, 'C25': 105, 'C26': 142572}
    # print(df.head(5))
    # print(feat_sizes)

    train, test = train_test_split(df, test_size=0.2, random_state=2021)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    local_rank = int(os.environ["LOCAL_RANK"])
    model = deepfm(feat_sizes, sparse_feature_columns=sparse_features, dense_feature_columns=dense_features,
                   dnn_hidden_units=[1000, 500, 250], dnn_dropout=0.9, ebedding_size=16,
                   l2_reg_linear=1e-3, device=device).to(local_rank)
    ddp_model = DDP(model, [local_rank])
    loss_fn = nn.BCELoss(reduction='mean')
    # optimiz er = optim.SGD(ddp_model.parameters(), lr=0.001)
    global optimizer
    global ckp_path
    optimizer = optim.Adam(ddp_model.parameters(), lr=lr, weight_decay=wd)
    ckp_path = "checkpoint.pt"
    if os.path.exists(ckp_path):
        print(f"load checkpoint from {ckp_path}")
        checkpoint = load_checkpoint(ckp_path)
        ddp_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimize_state_dict"])
        first_epoch = checkpoint["epoch"]
        del checkpoint

    train_label = pd.DataFrame(train['label'])
    train_data = train.drop(columns=['label'])
    # print(train.head(5))
    train_tensor_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(train_data)),
                                                       torch.from_numpy(np.array(train_label)))
    train_loader = DataLoader(dataset=train_tensor_data, shuffle=True, batch_size=batch_size)

    test_label = pd.DataFrame(test['label'])
    test_data = test.drop(columns=['label'])
    test_tensor_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(test_data)),
                                                      torch.from_numpy(np.array(test_label)))
    test_loader = DataLoader(dataset=test_tensor_data, shuffle=False, batch_size=batch_size)

    auc = get_auc(test_loader, ddp_model.to(device))
    print('test auc: {:.3f}'.format(auc))
    dist.destroy_process_group()
