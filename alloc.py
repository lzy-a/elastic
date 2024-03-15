import requests
import json

# ClickHouse服务器的URL
url = ' http://themis-olap-gateway.internal/'

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
data = {'query': query}

# 发送POST请求
response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

# 检查响应状态码
if response.status_code == 200:
    # 解析并打印结果
    result = response.json()
    print(result)
else:
    print(f"请求失败，状态码：{response.status_code}")