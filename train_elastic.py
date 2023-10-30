import argparse
import os
import shutil
import socket

import pandas as pd
import multiprocessing
from multiprocessing import Process, Value
import sys
import time
from datetime import timedelta
from datetime import datetime
import tempfile
from urllib.parse import urlparse
import json
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss, roc_auc_score

from torch.nn.parallel import DistributedDataParallel as DDP
from kafka import KafkaConsumer
from kafka import KafkaAdminClient
from prometheus_client import Gauge
from prometheus_client import start_http_server
import signal
import threading
import threading
import queue
from multiprocessing import Process, Queue

from deepfm import deepfm

auc_g = Gauge('auc', 'auc')
throughput_g = Gauge('throughput', 'samples per sec')
lag_g = Gauge('lag', 'kafka lag')
loss_g = Gauge('loss', 'loss')
save_g = Gauge('save', 'save cost time')
get_item_g = Gauge('get_item', 'read a sample cost time')
get_sample_g = Gauge('get_sample', 'read samples cost time')
grad_span_g = Gauge('grad', 'grad cost time')
sync_span_g = Gauge('sync', 'sync cost time')
lag_g.set(0)
# 设置 Kafka 主题和服务器地址
bootstrap_servers = '11.32.251.131:9092,11.32.224.11:9092,11.32.218.18:9092'
topic = 'stream8'
group = '1'
client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
# 创建 Kafka 消费者
num_consumers = 4
full_cnt = 0
empty_cnt = 0
# consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers, group_id=group, auto_offset_reset='latest')
# consumer.subscribe([topic])

global_batch_size = 1024


class DCAPDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 10 ** 8

    def __getitem__(self, idx):
        start = time.time()
        local_rank = int(os.environ["LOCAL_RANK"])
        message = next(consumer)
        message_dict = json.loads(message.value.decode('utf-8'))

        # 现在你可以通过键来访问train和label数据
        train_data = message_dict['train']
        label_data = message_dict['label']

        train_tensor = torch.tensor(list(train_data.values())).float().cuda(local_rank)
        label_tensor = torch.tensor(list(label_data.values())).float().cuda(local_rank)

        timestamp = message.timestamp
        get_item_g.set(time.time() - start)
        return {
            'input_data': train_tensor,
            'labels': label_tensor,
            'timestamp': timestamp
        }


# class DeepfmDataset(torch.utils.data.Dataset):
#     def __init__(self, buffer_size=5000, num_consumers=4):
#         self.buffer_size = buffer_size
#         self.buffer = []
#         self.local_rank = int(os.environ["LOCAL_RANK"])
#         self.num_consumers = num_consumers
#         self.consumer = consumer
#         self.refill_buffer()
#
#
#
#     def refill_buffer(self):
#         start = time.time()
#         while True:
#             while len(self.buffer) < self.buffer_size:
#                 message = next(self.consumer)
#                 message_dict = json.loads(message.value.decode('utf-8'))
#
#                 # 现在你可以通过键来访问train和label数据
#                 train_data = message_dict['train']
#                 label_data = message_dict['label']
#
#                 train_tensor = torch.tensor(list(train_data.values())).float().cuda(self.local_rank)
#                 label_tensor = torch.tensor(list(label_data.values())).float().cuda(self.local_rank)
#
#                 timestamp = message.timestamp
#                 get_item_g.set(time.time() - start)
#
#                 self.buffer.append({
#                     'input_data': train_tensor,
#                     'labels': label_tensor,
#                     'timestamp': timestamp
#                 })
#
#     def __len__(self):
#         return 10 ** 5
#
#     def __getitem__(self, idx):
#         if len(self.buffer) == 0:
#             self.refill_buffer()
#         return self.buffer.pop(0)
class DeepfmDataset(torch.utils.data.Dataset):
    def __init__(self, buffer_size=50000, num_consumers=8):
        self.buffer_size = buffer_size
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.num_consumers = num_consumers
        self.consumer_queues = [queue.Queue() for _ in range(num_consumers)]

        # Start Kafka consumers in separate threads
        self.consumer_threads = []
        for i in range(num_consumers):
            thread = threading.Thread(target=self.kafka_consumer, args=(i, topic))
            thread.start()
            self.consumer_threads.append(thread)

    def kafka_consumer(self, consumer_id, topic):
        consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers, group_id=group, auto_offset_reset='latest')
        consumer.subscribe([topic])  # 订阅主题
        # 分配分区并加入消费者组
        assigned_partitions = consumer.assignment()
        consumer.commit()
        print(f"Consumer {consumer_id} initialized and assigned to partitions: {assigned_partitions}")
        while True:
            while len(self.buffer) < self.buffer_size:
                start = time.time()
                message = next(consumer)
                if message is not None:
                    message_dict = json.loads(message.value.decode('utf-8'))
                    train_data = message_dict['train']
                    label_data = message_dict['label']
                    train_tensor = torch.tensor(list(train_data.values())).float().cuda(self.local_rank)
                    label_tensor = torch.tensor(list(label_data.values())).float().cuda(self.local_rank)
                    timestamp = message.timestamp
                    get_item_g.set(time.time() - start)
                    with self.buffer_lock:
                        self.buffer.append({
                            'input_data': train_tensor,
                            'labels': label_tensor,
                            'timestamp': timestamp
                        })
            # print("buffer full")
            global full_cnt
            full_cnt = full_cnt + 1

    def __len__(self):
        return 10 ** 5

    def __getitem__(self, idx):
        start = time.time()
        while len(self.buffer) == 0:
            # 等待一段时间，然后重试
            # print("sleep 0.01 for buffer refill")
            global empty_cnt
            empty_cnt = empty_cnt + 1
            time.sleep(0.01)  # 0.1秒的等待时间，你可以根据需要调整

        with self.buffer_lock:
            data = self.buffer.pop(0)
        get_item_g.set(time.time() - start)
        return data


# 定义自定义数据加载器
class KafkaDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 10 ** 8

    def __getitem__(self, idx):
        start = time.time()
        local_rank = int(os.environ["LOCAL_RANK"])
        message = next(consumer)
        data = message.value.decode('utf-8').split(',')
        input_data = torch.tensor([float(d) for d in data[:10]]).cuda(local_rank)
        labels = torch.tensor([float(d) for d in data[10:]]).cuda(local_rank)
        timestamp = message.timestamp
        get_item_g.set(time.time() - start)
        return {
            'input_data': input_data,
            'labels': labels,
            'timestamp': timestamp
        }


def save_checkpoint(epoch, model, optimizer, path):
    # 创建一个临时文件路径
    if int(os.environ["LOCAL_RANK"]) != 0:
        return
    tmp_path = path + ".tmp"

    # 首先将模型保存到临时文件中
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimize_state_dict": optimizer.state_dict(),
    }, tmp_path)

    if os.path.exists(tmp_path):
        # 然后将临时文件移动到目标文件
        shutil.move(tmp_path, path)


def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint


def get_auc(loader, model):
    pred, target = [], []
    model.eval()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = "cuda:{}".format(local_rank)
    with torch.no_grad():
        for sample in loader:
            x = sample["input_data"]
            y = sample["labels"]
            y_hat = model(x).to(device)
            pred += list(y_hat.cpu().numpy())  # 将y_hat移动到CPU上
            target += list(y.cpu().numpy())  # 将y移动到CPU上
    auc = roc_auc_score(target, pred)
    auc_g.set(auc)
    return auc


def train():
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    col_names = ['label'] + dense_features + sparse_features
    # df = pd.read_csv('data/test.txt', names=col_names, sep='\t')
    lr = 0.001
    wd = 0.00001
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) train worker starting...")
    # model = ToyModel().cuda(local_rank)
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
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    model = deepfm(feat_sizes=feat_sizes, sparse_feature_columns=sparse_features, dense_feature_columns=dense_features,
                   dnn_hidden_units=[1000, 500, 250], dnn_dropout=0.9, ebedding_size=16,
                   l2_reg_linear=1e-3, device=f"cuda:{local_rank}").to(local_rank)
    ddp_model = DDP(model, [local_rank])
    loss_fn = nn.BCELoss(reduction='mean')
    optimizer = optim.Adam(ddp_model.parameters(), lr=lr, weight_decay=wd)
    ckp_path = "checkpoint.pt"
    if os.path.exists(ckp_path):
        checkpoint = torch.load(ckp_path, map_location="cpu")
        ddp_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimize_state_dict"])
        del checkpoint

    # sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size,
    #                                                           rank=rank)
    dataset = DeepfmDataset(num_consumers=num_consumers)
    dataloader = DataLoader(dataset, batch_size=global_batch_size)

    global i
    i = 0
    step = 0
    step_timer = time.time()
    auc_timer = time.time()
    model.train().to(local_rank)
    while True:
        start = time.time()
        for sample in dataloader:
            input_data = sample["input_data"]
            labels = sample["labels"]
            get_sample_time = time.time() - start
            get_sample_g.set(get_sample_time)
            timestamp = sample["timestamp"][0].item() / 1000
            print(f"[{os.getpid()}] Received input data: {input_data}")
            print(f"[{os.getpid()}] Received labels: {labels}")
            lag = time.time() - timestamp
            lag_g.set(lag)
            start = time.time()
            optimizer.zero_grad()
            outputs = ddp_model(input_data.to(local_rank))  # 输入数据要进行维度扩展
            loss = loss_fn(outputs, labels.to(local_rank))
            loss.backward()
            grad_span_g.set(time.time() - start)
            print(f"[{os.getpid()}] epoch {i} (rank = {rank}, local_rank = {local_rank}) loss = {loss.item()}\n")
            step = step + 1
            if step == 10:
                throughput_g.set(10 * int(os.environ["WORLD_SIZE"]) * global_batch_size / (time.time() - step_timer))
                step = 0
                step_timer = time.time()
            loss_g.set(loss.item())
            start = time.time()
            optimizer.step()
            sync_span_g.set(time.time() - start)
            if i % 500 == 99:
                start = time.time()
                print(f'empty_cnt {empty_cnt} full_cnt {full_cnt}')
                save_checkpoint(i, ddp_model, optimizer, ckp_path)
                save_g.set(time.time() - start)
            start = time.time()
            if time.time() - auc_timer > 180:
                auc = get_auc(dataloader, ddp_model)
                auc_g.set(auc)
                auc_timer = time.time()
            i += 1


# 先初始化好kafka再dist init
def kafka_setup():
    ws = os.environ["WORLD_SIZE"]
    member_count = 0
    while member_count < int(ws) * num_consumers:
        group_description = client.describe_consumer_groups([group])
        print(group_description)
        for group_des in group_description:
            if group_des.group != group or group_des.state != 'Stable':
                continue
            else:
                member_count = len(group_des.members)
                break
        print(f"[{os.getpid()}] consumer cnt {member_count} ws {ws} total {ws * num_consumers}")
        time.sleep(0.1)


def run():
    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)
    if int(os.environ["RANK"]) == 0:
        start_http_server(8000)  # prom exporter http://$pod_ip:8000/metrics

    # kafka_setup()
    os.environ["MASTER_ADDR"] = socket.gethostbyname('elastic-master-service.default.svc.cluster.local')
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    start = time.time()
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=15))
    print(f"[{os.getpid()}] init time: {time.time() - start}")
    train()
    dist.destroy_process_group()


# def signal_handler(sig, frame):
#     print('Signal received, saving checkpoint...')
#     save_checkpoint(i, ddp_model, optimizer, ckp_path)
#     sys.exit(0)


if __name__ == "__main__":
    run()
