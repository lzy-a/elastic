from kafka import KafkaProducer
import time
import random

bootstrap_servers = '11.32.251.131:9092,11.32.224.11:9092,11.32.218.18:9092'
topic = 'stream-6'

producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

fast_rate = 0.001
slow_rate = 0.01


def send_message(sleep_interval):
    input_data = ','.join([str(random.uniform(0, 1)) for _ in range(10)])
    labels = ','.join([str(random.uniform(0, 1)) for _ in range(5)])

    message = "{},{}".format(input_data, labels).encode('utf-8')
    producer.send(topic, value=message)
    print("Sent message: {}".format(message))
    time.sleep(sleep_interval)


while True:
    start_time = time.time()
    while time.time() - start_time < 120:  # 快速生产数据的阶段，持续2分钟
        interval = fast_rate + random.uniform(-0.0002, 0.0002)
        send_message(interval)

    start_time = time.time()
    while time.time() - start_time < 60:  # 慢速生产数据的阶段，持续1分钟
        interval = slow_rate + random.uniform(-0.002, 0.002)
        send_message(interval)
