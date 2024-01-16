from kafka import KafkaProducer
import time
import random
from prometheus_client import Gauge
from prometheus_client import start_http_server
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import OrderedDict, namedtuple, defaultdict
import json
import multiprocessing


bootstrap_servers = '11.32.251.131:9092,11.32.224.11:9092,11.32.218.18:9092'
topic = 'stream16'

producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

cnt = 0
g = Gauge('rate', 'kafka produce samples per sec')
start_http_server(8000)  # prom exporter http://$pod_ip:8000/metrics

sparse_feature = ['C' + str(i) for i in range(1, 27)]
dense_feature = ['I' + str(i) for i in range(1, 14)]
col_names = ['label'] + dense_feature + sparse_feature


def preprocess_data(chunk):
    chunk[sparse_feature] = chunk[sparse_feature].fillna('-1')
    chunk[dense_feature] = chunk[dense_feature].fillna('0')
    for feat in sparse_feature:
        lbe = LabelEncoder()
        chunk[feat] = lbe.fit_transform(chunk[feat])
    nms = MinMaxScaler(feature_range=(0, 1))
    chunk[dense_feature] = nms.fit_transform(chunk[dense_feature])
    return chunk


def process_data(chunk):
    preprocess_data(chunk)
    train_label = pd.DataFrame(chunk['label'])
    train = chunk.drop(columns=['label'])
    for train_row, label_row in zip(train.iterrows(), train_label.iterrows()):
        train_data = train_row[1]
        label_data = label_row[1]
        message_dict = {"train": train_data.to_dict(), "label": label_data.to_dict()}
        message = json.dumps(message_dict).encode('utf-8')
        send_message(message, False)


def send_message(message, p):
    global cnt
    cnt += 1
    producer.send(topic, value=message)
    if p:
        print("Sent message: {}".format(message))


if __name__ == '__main__':
    target_rate = 3000  # 目标速率每秒3000个消息
    while True:
        start = time.time()
        reader = pd.read_csv('./data/dac_sample.txt', names=col_names, sep='\t', chunksize=target_rate)
        end = time.time()
        span = end - start
        start = end
        # print(f"reader {span}")
        control_timer = time.time()
        for data in reader:
            end = time.time()
            span = end - start
            start = end
            print(f"read {span}")
            process_data(data)
            chunk_time = time.time() - control_timer
            sleep_time = max(0.0, 1.0 - chunk_time)
            g.set(target_rate / (chunk_time + sleep_time))
            time.sleep(sleep_time)
            print(f"throughput {target_rate / (chunk_time + sleep_time)}")
            control_timer = time.time()
