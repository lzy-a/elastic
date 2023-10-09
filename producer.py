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
bootstrap_servers = '11.32.251.131:9092,11.32.224.11:9092,11.32.218.18:9092'
topic = 'stream-6'

producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

fast_rate = 0.001
slow_rate = 0.01

g = Gauge('rate', 'kafka produce samples per sec')
g.set(1000)
start_http_server(8000)  # prom exporter http://$pod_ip:8000/metrics


def send_message(message, sleep_interval):
    producer.send(topic, value=message)
    print("Sent message: {}".format(message))
    sleep_interval = sleep_interval + random.uniform(-0.3*sleep_interval, 0.3*sleep_interval)
    g.set(1.0/sleep_interval)
    time.sleep(sleep_interval)


if __name__ == '__main__':
    batch_size = 1024
    lr = 1e-3
    wd = 0
    epoches = 100
    seed = 2022
    embedding_size = 16
    device = 'cuda:0'

    sparse_feature = ['C' + str(i) for i in range(1, 27)]
    dense_feature = ['I' + str(i) for i in range(1, 14)]
    col_names = ['label'] + dense_feature + sparse_feature

    data = pd.read_csv('./data/dac_sample.txt', names=col_names, sep='\t')

    data[sparse_feature] = data[sparse_feature].fillna('-1', )
    data[dense_feature] = data[dense_feature].fillna('0', )
    target = ['label']

    feat_sizes = {}
    feat_sizes_dense = {feat: 1 for feat in dense_feature}
    feat_sizes_sparse = {feat: len(data[feat].unique()) for feat in sparse_feature}
    feat_sizes.update(feat_sizes_dense)
    feat_sizes.update(feat_sizes_sparse)
    print(feat_sizes)
#{'I1': 1, 'I2': 1, 'I3': 1, 'I4': 1, 'I5': 1, 'I6': 1, 'I7': 1, 'I8': 1, 'I9': 1, 'I10': 1, 'I11': 1, 'I12': 1, 'I13': 1, 'C1': 541, 'C2': 497, 'C3': 43870, 'C4': 25184, 'C5': 145, 'C6': 12, 'C7': 7623, 'C8': 257, 'C9': 3, 'C10': 10997, 'C11': 3799, 'C12': 41312, 'C13': 2796, 'C14': 26, 'C15': 5238, 'C16': 34617, 'C17': 10, 'C18': 2548, 'C19': 1303, 'C20': 4, 'C21': 38618, 'C22': 11, 'C23': 14, 'C24': 12335, 'C25': 51, 'C26': 9527}

    for feat in sparse_feature:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    nms = MinMaxScaler(feature_range=(0, 1))
    data[dense_feature] = nms.fit_transform(data[dense_feature])

    # fixlen_feature_columns = [(feat, 'sparse') for feat in sparse_feature] + [(feat, 'dense') for feat in dense_feature]
    # dnn_feature_columns = fixlen_feature_columns
    # linear_feature_columns = fixlen_feature_columns
    # print(fixlen_feature_columns)
#[('C1', 'sparse'), ('C2', 'sparse'), ('C3', 'sparse'), ('C4', 'sparse'), ('C5', 'sparse'), ('C6', 'sparse'), ('C7', 'sparse'), ('C8', 'sparse'), ('C9', 'sparse'), ('C10', 'sparse'), ('C11', 'sparse'), ('C12', 'sparse'), ('C13', 'sparse'), ('C14', 'sparse'), ('C15', 'sparse'), ('C16', 'sparse'), ('C17', 'sparse'), ('C18', 'sparse'), ('C19', 'sparse'), ('C20', 'sparse'), ('C21', 'sparse'), ('C22', 'sparse'), ('C23', 'sparse'), ('C24', 'sparse'), ('C25', 'sparse'), ('C26', 'sparse'), ('I1', 'dense'), ('I2', 'dense'), ('I3', 'dense'), ('I4', 'dense'), ('I5', 'dense'), ('I6', 'dense'), ('I7', 'dense'), ('I8', 'dense'), ('I9', 'dense'), ('I10', 'dense'), ('I11', 'dense'), ('I12', 'dense'), ('I13', 'dense')]

    # train, test = train_test_split(data, test_size=0.2, random_state=seed)
    train = data
    # device = 'cuda:0'
    # model = DCAP(feat_sizes, embedding_size, linear_feature_columns, dnn_feature_columns).to(device)

    train_label = pd.DataFrame(train['label'])
    train = train.drop(columns=['label'])

    interval = fast_rate
    while True:
        if interval == fast_rate:
            interval = slow_rate
        else:
            interval = fast_rate
        for train_row, label_row in zip(train.iterrows(), train_label.iterrows()):
            train_data = train_row[1]
            label_data = label_row[1]
            message_dict = {"train": train_data.to_dict(), "label": label_data.to_dict()}
            message = json.dumps(message_dict).encode('utf-8')
            # message = "{},{}".format(train_data, label_data).encode('utf-8')
            send_message(message, fast_rate)
            print("Sent message: {}".format(message))

