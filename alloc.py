import requests

# ClickHouse服务器的URL
url = ' http://themis-olap-gateway.internal/'

token = 'Ch1saXV6aXlhbmcwNS91c2VyQGt1YWlzaG91LmNvbRoNMTcyLjIzLjg2LjExMSIEMTcxOSjrvvOk5TEw64mD6OYxQAJKFnsiaXAiOiIxNzIuMjMuODYuMTExIn0.KLzXx2WDpGn6opn8cIvTz6N9J7Q9iOxln6wE-H2C8yw'
headers = {
    'Authorization': f'Bearer {token}',
    # 可能还有其他需要的头部信息
}

# SQL查询
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

# 构造请求体
data = query.encode('utf-8')

try:
    # 发送POST请求
    response = requests.post(url, data=data, headers=headers)

    # 检查响应状态码
    if response.status_code == 200:
        # 解析并打印结果
        result = response.text
        print(result)
    else:
        print(f"请求失败，状态码：{response.status_code}")
        print(f"错误信息：{response.text}")
except requests.exceptions.RequestException as e:
    print(f"请求异常：{e}")
