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
# from kafka import KafkaConsumer
from kafka import KafkaAdminClient
from prometheus_client import Gauge
from prometheus_client import start_http_server
from confluent_kafka import Consumer


def on_assign(consumer, partitions):
    print('Assignment:', partitions)
    ws = os.environ["WORLD_SIZE"]
    member_count = 0
    while member_count < int(ws):
        group_description = client.describe_consumer_groups([group])
        print(group_description)
        for group_des in group_description:
            if group_des.group != group or group_des.state != 'Stable':
                continue
            else:
                member_count = len(group_des.members)
                break
        if member_count == ws:
            break
        print(f"[{os.getpid()}] consumer cnt {member_count} ws {ws}")
        msg = consumer.poll(1.0)
        time.sleep(0.1)


g = Gauge('lag', 'kafka lag')
g.set(0)
# 设置 Kafka 主题和服务器地址
bootstrap_servers = '11.32.251.131:9092,11.32.224.11:9092,11.32.218.18:9092'
topic = 'stream-6'
group = '1'
conf = {
    'bootstrap.servers': bootstrap_servers,
    'group.id': group,
    'auto.offset.reset': 'latest'
}
client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
# 创建 Kafka 消费者
consumer = Consumer(conf)
consumer.subscribe([topic], on_assign=on_assign)
lag_file = open('lag.txt', 'a')
proc_file = open('proc.txt', 'w')
global_batch_size = 12


# 定义自定义数据加载器
class KafkaDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 10 ** 8

    def __getitem__(self, idx):
        start = time.time()
        local_rank = int(os.environ["LOCAL_RANK"])
        message = consumer.poll(1.0)
        while message is None:
            message = consumer.poll(1.0)
        data = message.value.decode('utf-8').split(',')
        input_data = torch.tensor([float(d) for d in data[:10]]).cuda(local_rank)
        labels = torch.tensor([float(d) for d in data[10:]]).cuda(local_rank)
        timestamp = message.timestamp
        proc_file.write(f"[{os.getpid()}] get-item-span = {time.time() - start}\n")
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
    batch_size = int(global_batch_size / world_size)
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
            g.set(lag)
            lag_file.write(f"epoch:{i}, Lag: {lag}\n")
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


def run():
    if int(os.environ["RANK"]) == 0:
        start_http_server(8000)  # prom exporter http://$pod_ip:8000/metrics
    # kafka_setup()
    os.environ["MASTER_ADDR"] = socket.gethostbyname('elastic-master-service.default.svc.cluster.local')
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    train()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
