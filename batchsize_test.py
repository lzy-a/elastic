import shutil
import socket

import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.profiler import profile, record_function, ProfilerActivity

import sys
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from deepfm import deepfm
import time
from prometheus_client import Gauge
from prometheus_client import start_http_server
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import csv
import torchvision.models as models
from test_model_for_batchsize.resnet import ResNet, BasicBlock


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
    auc_g.set(auc)
    return auc


def save_checkpoint(epoch, model, optimizer, path):
    # 创建一个临时文件路径
    if int(os.environ["LOCAL_RANK"]) != 0:
        return
    tmp_path = path + ".tmp"

    # 首先将模型保存到临时文件中
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimize_state_dict": optimizer.state_dict(),
    }, tmp_path)

    if os.path.exists(tmp_path):
        # 然后将临时文件移动到目标文件
        shutil.move(tmp_path, path)


def setup_environment():
    os.environ["MASTER_ADDR"] = socket.gethostbyname('elastic-master-service.default.svc.cluster.local')
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        start_http_server(8000)


def prepare_data(seed=1024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    col_names = ['label'] + dense_features + sparse_features
    df = pd.read_csv('data/small.txt', names=col_names, sep='\t')
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
    #               'I12': 1, 'I13': 1, 'C1': 1460, 'C2': 583, 'C3': 10131227, 'C4': 2202608, 'C5': 305, 'C6': 24,
    #               'C7': 12517, 'C8': 633, 'C9': 3, 'C10': 93145, 'C11': 5683, 'C12': 8351593, 'C13': 3194, 'C14': 27,
    #               'C15': 14992, 'C16': 5461306, 'C17': 10, 'C18': 5652, 'C19': 2173, 'C20': 4, 'C21': 7046547,
    #               'C22': 18, 'C23': 15, 'C24': 286181, 'C25': 105, 'C26': 142572}

    return df, feature_names, dense_features, sparse_features, feat_sizes


def get_loader(df, feature_names, batch_size=10240):
    train, test = train_test_split(df, test_size=0.2, random_state=2021)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    train_label = pd.DataFrame(train['label'])
    train_data = train.drop(columns=['label'])
    train_tensor_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(train_data)),
                                                       torch.from_numpy(np.array(train_label)))
    sampler = DistributedSampler(train_tensor_data)
    train_loader = DataLoader(dataset=train_tensor_data, batch_size=batch_size, sampler=sampler)

    test_label = pd.DataFrame(test['label'])
    test_data = test.drop(columns=['label'])
    test_tensor_data = torch.utils.data.TensorDataset(torch.from_numpy(np.array(test_data)),
                                                      torch.from_numpy(np.array(test_label)))
    test_loader = DataLoader(dataset=test_tensor_data, shuffle=False, batch_size=batch_size, num_workers=1)
    return train_loader, test_loader


if __name__ == "__main__":
    setup_environment()
    local_rank = int(os.environ["LOCAL_RANK"])
    loss_g = Gauge('loss', 'loss')
    auc_g = Gauge('auc', 'auc')

    # batch_size = 10240
    lr = 0.0005
    wd = 0.0001
    epoches = 10
    # prepare data
    seed = 1024
    df, feature_names, dense_features, sparse_features, feat_sizes = prepare_data(seed)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "cuda:{}".format(local_rank)
    # model = deepfm(feat_sizes, sparse_feature_columns=sparse_features, dense_feature_columns=dense_features,
    #                dnn_hidden_units=[1000, 500, 250], dnn_dropout=0.9, ebedding_size=16,
    #                l2_reg_linear=1e-3, device=device).to(local_rank)
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model = DDP(model, [local_rank])

    # train_loader, test_loader = get_loader(df, feature_names, batch_size=batch_size)

    # loss_func = nn.BCELoss(reduction='mean')
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # prof.export_chrome_trace("trace.json")

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    # CSV file setup
    csv_file_path = "experiment_results.csv"
    fieldnames = ["BatchSize", "DataTime", "ForwardTime", "LossTime", "OptimizerTime", "StepTime",
                  "Throughput"]

    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        for batch_size in batch_sizes:
            print(f'batch_size:{batch_size}')
            # Update batch size
            train_loader, _ = ResNet.get_loader(batch_size=batch_size)

            start = time.time()
            model_total = 0
            loss_total = 0
            step_total = 0
            optimizer_total = 0
            data_total = 0
            total_tmp = 0

            for epoch in range(epoches):
                total_loss_epoch = 0.0

                model.train().to(device)
                step_start = time.time()
                data_start = time.time()
                for index, (x, y) in enumerate(train_loader):
                    x = x.to(device).float()
                    y = y.to(device).float()
                    data_time = time.time() - data_start
                    data_total += data_time
                    with record_function("Forward Pass"):
                        model_start = time.time()
                        y_hat = model(x).to(device)
                        # torch.cuda.synchronize()
                        model_time = time.time() - model_start
                        model_total += model_time

                    with record_function("Loss and Backward Pass"):
                        loss_start = time.time()
                        optimizer.zero_grad()
                        loss = loss_func(y_hat, y)
                        loss.backward()
                        # torch.cuda.synchronize()
                        loss_time = time.time() - loss_start
                        loss_total += loss_time

                    with record_function("Optimizer Pass"):
                        optimizer_start = time.time()
                        optimizer.step()
                        # torch.cuda.synchronize()
                        optimizer_time = time.time() - optimizer_start
                        optimizer_total += optimizer_time

                    total_tmp += 1
                    step_time = time.time() - step_start
                    step_total += step_time
                    data_start = time.time()
                    step_start = time.time()
                #
                # save_checkpoint(epoch, model, optimizer, "ddp_ckp.pt")
                # auc = get_auc(test_loader, model.to(device))
                # auc_g.set(auc)
                # print('epoch/epoches: {}/{}, train loss: {:.3f}, test auc: {:.3f}'.format(epoch, epoches,
                #                                                                           total_loss_epoch / total_tmp, auc))
                print(
                    'epoch/epoches: {}/{}, data time: {:.5f},forward time: {:.5f},loss time: {:.5f},optimizer time: {:.5f}, step time: {:.5f}'.format(
                        epoch, epoches, data_total / total_tmp, model_total / total_tmp,
                                        loss_total / total_tmp, optimizer_total / total_tmp,
                                        step_total / total_tmp))
                if epoch == 0:
                    model_total = 0
                    loss_total = 0
                    step_total = 0
                    optimizer_total = 0
                    data_total = 0
                    total_tmp = 0
            # Record data for the current experiment
            experiment_data = {
                "BatchSize": batch_size,
                "DataTime": data_total / total_tmp,
                "ForwardTime": model_total / total_tmp,
                "LossTime": loss_total / total_tmp,
                "OptimizerTime": optimizer_total / total_tmp,
                "StepTime": step_total / total_tmp,
                "Throughput": batch_size / (step_total / total_tmp),
            }

            # Write the data to the CSV file
            csv_writer.writerow(experiment_data)

    dist.destroy_process_group()
