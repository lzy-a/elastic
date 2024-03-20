import requests
import pandas as pd
import csv


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
    token = 'Ch1saXV6aXlhbmcwNS91c2VyQGt1YWlzaG91LmNvbRoNMTcyLjI1LjM2LjEwNSjqh-vU5TEw6sGi2OUxOAo.27aANT0SkqyAs1sdw3yU4nsCMHKjFxwWIKCBBDl30WM'
    clickhouse_client = ClickHouseQuery(' http://themis-olap-gateway.internal/',
                                        token=token,
                                        user='liuziyang05')
    result = clickhouse_client.execute_query(query)
    if result is not None:
        print(result)
        clickhouse_client.res_to_csv(result, './dataset/predict_data.csv')
