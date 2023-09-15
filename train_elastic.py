import argparse
import os
import socket
import sys
import time
from datetime import datetime
import tempfile
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
from kafka import KafkaConsumer
from kafka import KafkaAdminClient

# 设置 Kafka 主题和服务器地址
bootstrap_servers = '11.32.251.131:9092,11.32.224.11:9092,11.32.218.18:9092'
topic = 'stream-6'
group = '1'
client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
# 创建 Kafka 消费者
consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers, group_id=group, auto_offset_reset='latest')
consumer.subscribe([topic])
lag_file = open('lag.txt', 'w')
proc_file = open('proc.txt', 'w')


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
        proc_file.write(f"[{os.getpid()}]-------get-item-span = {time.time() - start}\n")
        proc_file.flush()
        return {
            'input_data': input_data,
            'labels': labels,
            'timestamp': timestamp
        }


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def save_checkpoint(epoch, model, optimizer, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimize_state_dict": optimizer.state_dict(),
    }, path)


def load_checkpoint(path):
    checkpoint = torch.load(path)
    return checkpoint


def train():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) train worker starting...")
    model = ToyModel().cuda(local_rank)
    ddp_model = DDP(model, [local_rank])
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    ckp_path = "checkpoint.pt"
    if os.path.exists(ckp_path):
        print(f"load checkpoint from {ckp_path}")
        checkpoint = load_checkpoint(ckp_path)
        ddp_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimize_state_dict"])
        first_epoch = checkpoint["epoch"]

    # 创建数据加载器
    batch_size = 2
    dataset = KafkaDataset()
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size,
    #                                                           rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    i = 0
    while True:
        for sample in dataloader:
            input_data = sample["input_data"]
            labels = sample["labels"]
            timestamp = sample["timestamp"][0].item() / 1000
            print(f"[{os.getpid()}] Received input data: {input_data}")
            print(f"[{os.getpid()}] Received labels: {labels}")
            lag = time.time() - timestamp
            lag_file.write(f"cur: {time.time()}, timestamp: {timestamp}, Lag: {lag}\n")
            lag_file.flush()
            start = time.time()
            optimizer.zero_grad()
            outputs = ddp_model(input_data)  # 输入数据要进行维度扩展
            loss = loss_fn(outputs, labels)
            loss.backward()
            proc_file.write(f"epoch {i} grad-span = {time.time() - start}\n")
            proc_file.flush()
            print(f"[{os.getpid()}] epoch {i} (rank = {rank}, local_rank = {local_rank}) loss = {loss.item()}\n")
            start = time.time()
            optimizer.step()
            proc_file.write(f"epoch {i} sync-span = {time.time() - start}\n")
            proc_file.flush()
            # if i % 10 == 0:
            # lag_file.flush()
            # proc_file.flush()
            save_checkpoint(i, ddp_model, optimizer, ckp_path)
            i += 1


# 不要在Kafka消费者组初始化完成之前进入训练过程
def kafka_warmup():
    # 订阅主题并加入消费者组
    start = time.time()
    while time.time() - start < 10:
        time.sleep(1)
        msg = consumer.poll(timeout_ms=1000, max_records=1)


# 先初始化好kafka再dist init
def kafka_setup():
    ws = os.environ["WORLD_SIZE"]
    group_description = client.describe_consumer_groups([group])
    print(group_description)
    # member_count = len(group_description[group].members)
    # while member_count < int(ws):
    #     print(f"[{os.getpid()}] consumer cnt {member_count} ws {ws}")
    #     msg = consumer.poll(timeout_ms=1000, max_records=1)
    #     group_description = client.describe_consumer_groups([group])
    #     member_count = len(group_description[group].members)


def run():
    kafka_setup()
    os.environ["MASTER_ADDR"] = socket.gethostbyname('elastic-master-service.default.svc.cluster.local')
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    # kafka_warmup()
    train()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
