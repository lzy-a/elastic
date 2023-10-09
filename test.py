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

fast_rate = 0.001
slow_rate = 0.01

sparse_feature = ['C' + str(i) for i in range(1, 27)]
dense_feature = ['I' + str(i) for i in range(1, 14)]
col_names = ['label'] + dense_feature + sparse_feature
def send_message(message, sleep_interval):
    producer.send(topic, value=message)
    print("Sent message: {}".format(message))
    sleep_interval = sleep_interval + random.uniform(-0.3*sleep_interval, 0.3*sleep_interval)
    g.set(1.0/sleep_interval)
    time.sleep(sleep_interval)

def dosth(data):
    start = time.time()
    end = time.time()
    span = end - start
    start = end
    print(f"read {span}")
    data[sparse_feature] = data[sparse_feature].fillna('-1', )
    end = time.time()
    span = end - start
    start = end
    print(f"fillin sparse {span}")
    data[dense_feature] = data[dense_feature].fillna('0', )
    target = ['label']
    end = time.time()
    span = end - start
    start = end
    print(f"fillin dense {span}")
    for feat in sparse_feature:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    end = time.time()
    span = end - start
    start = end
    print(f"sparse trans {span}")
    nms = MinMaxScaler(feature_range=(0, 1))
    data[dense_feature] = nms.fit_transform(data[dense_feature])
    end = time.time()
    span = end - start
    start = end
    print(f"dense trans {span}")

    train = data

    train_label = pd.DataFrame(train['label'])
    train = train.drop(columns=['label'])
    end = time.time()
    span = end - start
    start = end
    print(f"drop {span}")

    for train_row, label_row in zip(train.iterrows(), train_label.iterrows()):
        train_data = train_row[1]
        label_data = label_row[1]
        message_dict = {"train": train_data.to_dict(), "label": label_data.to_dict()}
        message = json.dumps(message_dict).encode('utf-8')
        print("Sent message: {}".format(message))
        end = time.time()


if __name__ == '__main__':

    sparse_feature = ['C' + str(i) for i in range(1, 27)]
    dense_feature = ['I' + str(i) for i in range(1, 14)]
    col_names = ['label'] + dense_feature + sparse_feature

    while True:
        start = time.time()
        data = pd.read_csv('./data/dac_sample.txt', names=col_names, sep='\t',chunksize=1000)
        end = time.time()
        span = end - start
        start = end
        print(f"read {span}")
        dosth(data)