import requests
import pandas as pd
import csv
import aiflow


class ClickHouseQuery:
    def __init__(self, url, token, user):
        self.url = url
        self.token = token
        self.user = user
        self.headers = {
            'Ks-Auth-Principal': f'{user}/user@kuaishou.com',
            'Ks-Auth-Token': token,
            'Ks-Auth-Type': 'USER',
            'Ks-Auth-User': user,
            'Ks-Query-Id': '123456'
        }

    def execute_query(self, query):
        try:
            data = query.encode('utf-8')
            response = requests.post(self.url, data=data, headers=self.headers)

            if response.status_code == 200:
                return response.text
            else:
                print(f"请求失败，状态码：{response.status_code}")
                print(f"错误信息：{response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"请求异常：{e}")
            return None

    def res_to_csv(self, res, path):
        try:
            # 将字符串按行拆分成列表
            lines = res.strip().split('\n')

            # 去掉收尾行
            lines = lines[1:-1]

            # 初始化结果列表
            result = []

            # 每15行变成一行，第二个单元格是平均值
            for i in range(0, len(lines), 15):
                data = lines[i:i + 15]
                time_str = data[0].split('\t')[0]
                values = [float(row.split('\t')[1]) for row in data]
                avg_value = sum(values) / len(values)
                result.append([time_str, avg_value])

            # 将结果写入CSV文件
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['date', 'target'])
                writer.writerows(result)

            print(f"Result has been successfully written to {path}")
        except Exception as e:
            print(f"Error occurred while writing result to {path}: {e}")


# k1 = 3.4657635304789602 , k2 = 0.0009867267625321835 , k3 = 0.021788545751746453
class GPUAllocator:
    def __init__(self, k1=3.4657635304789602, k2=0.0009867267625321835, k3=0.021788545751746453):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

    def calculate_throughput(self, w):
        return 16384 / (self.k1 / w + self.k3 + self.k2 * w)

    def throughput_to_workernum(self, throughput):
        # 定义二分法搜索的范围
        left = 2
        right = 32

        # 二分搜索
        while left < right:
            mid = (left + right) // 2
            thrpt = self.calculate_throughput(mid)
            if thrpt >= throughput:
                right = mid
            else:
                left = mid + 1

        # 返回向上取整到最接近的 2 的倍数的结果
        return left if left % 2 == 0 else left + 1


class PredictionClient:
    def __init__(self, url):
        self.url = url

    def get_prediction(self):
        try:
            response = requests.get(self.url)
            if response.status_code == 200:
                data = response.json()
                if 'result' in data:
                    result_list = data['result']
                    float_array = [float(value[0]) for value in result_list]
                    return float_array
                else:
                    raise ValueError("Response does not contain 'result' field.")
            else:
                raise ValueError("Failed to fetch prediction. Status code: {}".format(response.status_code))
        except requests.exceptions.RequestException as e:
            print("Error: {}".format(e))
            return None


if __name__ == '__main__':

    # 使用示例
    query = """
    SELECT toStartOfMinute(fromUnixTimestamp(toInt64(ts / 1000))) AS minute,
           SUM(OneMinuteRate) AS sum_OneMinuteRate
    FROM ks_hdp.dataarch_kafka_quota_topic_metric
    WHERE cluster = 'wlf1-reco1'
      AND topic = 'produce_mini_reco_log'
      AND name = 'MessagesInPerSec'
      AND ts BETWEEN toUnixTimestamp(now() - toIntervalDay(1)) * 1000
      AND toUnixTimestamp(now()) * 1000
    GROUP BY toStartOfMinute(fromUnixTimestamp(toInt64(ts / 1000)))
    ORDER BY minute ASC
    """
    token = 'Ch1saXV6aXlhbmcwNS91c2VyQGt1YWlzaG91LmNvbRoOMTcyLjI1LjEwMC4xMDco-fDJqeYxMPmqga3mMTgK.MIS9K7sSWv3AZLhBwoh8BEz0X8RL7-NI9hSG3UJwZZY'
    clickhouse_client = ClickHouseQuery(' http://themis-olap-gateway.internal/',
                                        token=token,
                                        user='liuziyang05')
    result = clickhouse_client.execute_query(query)
    if result is not None:
        print(result)
        clickhouse_client.res_to_csv(result, './dataset/predict_data.csv')

    client = PredictionClient('http://127.0.0.1:5000/predict')
    prediction = client.get_prediction()
    print(f"Prediction: {prediction}")

    allocator = GPUAllocator()
    throughput = 190000  # 给定吞吐量
    worker_num = allocator.throughput_to_workernum(throughput)
    print(f"对应的 worker 数量为: {worker_num}")

    kml_controller = aiflow.KMLAIFlowController(28236)
    kml_controller.start()
    machine_num = worker_num / 2
    kml_controller.change_replicas('worker', machine_num)
    kml_controller.change_batch_size(16384 / machine_num)
    kml_controller.submit_sparse_config()
