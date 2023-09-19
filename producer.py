from kafka import KafkaProducer
import time
import random
from prometheus_client import Gauge
from prometheus_client import start_http_server

bootstrap_servers = '11.32.251.131:9092,11.32.224.11:9092,11.32.218.18:9092'
topic = 'stream-6'

producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

fast_rate = 0.001
slow_rate = 0.01

g = Gauge('rate', 'kafka produce samples per sec')
g.set(1000)
start_http_server(8000)  # prom exporter http://$pod_ip:8000/metrics


def send_message(sleep_interval):
    input_data = ','.join([str(random.uniform(0, 1)) for _ in range(10)])
    labels = ','.join([str(random.uniform(0, 1)) for _ in range(5)])

    message = "{},{}".format(input_data, labels).encode('utf-8')
    producer.send(topic, value=message)
    print("Sent message: {}".format(message))
    g.set(1.0/sleep_interval)
    time.sleep(sleep_interval)


while True:
    start_time = time.time()
    while time.time() - start_time < 120:  # 快速生产数据的阶段，持续2分钟
        interval = fast_rate + random.uniform(-0.0003, 0.0003)
        send_message(interval)

    start_time = time.time()
    while time.time() - start_time < 60:  # 慢速生产数据的阶段，持续1分钟
        interval = slow_rate + random.uniform(-0.003, 0.003)
        send_message(interval)
