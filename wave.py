from kafka import KafkaProducer
import time
import random
from prometheus_client import Gauge
from prometheus_client import start_http_server
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import json
import multiprocessing
from multiprocessing import Process, Manager, Value

bootstrap_servers = '11.32.251.131:9092,11.32.224.11:9092,11.32.218.18:9092'
topic = 'stream16'

# Define a multiprocessing manager for shared variables

g = Gauge('rate', 'kafka produce samples per sec')

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


def send_message(message, producer, p):
    with cnt.get_lock():
        cnt.value += 1
    producer.send(topic, value=message)
    if p:
        print("Sent message: {}".format(message))


def process_data(chunk, producer):
    preprocess_data(chunk)
    train_label = pd.DataFrame(chunk['label'])
    train = chunk.drop(columns=['label'])
    for train_row, label_row in zip(train.iterrows(), train_label.iterrows()):
        train_data = train_row[1]
        label_data = label_row[1]
        message_dict = {"train": train_data.to_dict(), "label": label_data.to_dict()}
        message = json.dumps(message_dict).encode('utf-8')
        send_message(message, producer, False)


def run_producer(producer_id, shared_target_rate, shared_throughput_dict):
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
    while True:
        start = time.time()
        target_rate = shared_target_rate.value
        reader = pd.read_csv('./data/dac_sample.txt', names=col_names, sep='\t', chunksize=target_rate)
        end = time.time()
        span = end - start
        start = end
        control_timer = time.time()
        for data in reader:
            end = time.time()
            span = end - start
            start = end
            process_data(data, producer)
            chunk_time = time.time() - control_timer
            sleep_time = max(0.0, 1.0 - chunk_time)
            with shared_throughput_dict.get_lock():
                shared_throughput_dict[producer_id] = target_rate / (chunk_time + sleep_time)
            time.sleep(sleep_time)
            print(f"throughput {target_rate / (chunk_time + sleep_time)}")
            control_timer = time.time()


if __name__ == '__main__':
    with Manager() as manager:
        target_rate = Value('i', 3000)  # 目标速率每秒3000个消息
        shared_throughput_dict = manager.dict()
        num_processes = 5

        # Start Prometheus HTTP server
        start_http_server(8000)

        processes = []
        for i in range(num_processes):
            process = Process(target=run_producer, args=(i, target_rate, shared_throughput_dict))
            processes.append(process)
            process.start()

        try:
            while True:
                total_throughput = sum(shared_throughput_dict.values())
                print(f"Main Process: Total Throughput: {total_throughput}")
                g.set(total_throughput)
                time.sleep(10)
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("Terminating all processes...")
            for process in processes:
                process.terminate()
