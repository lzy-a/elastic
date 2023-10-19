import numpy as np
import pandas as pd


# 定义全天的24小时
hours = np.arange(24)

# 设置一年日期范围
date_range = pd.date_range(start='1/1/2023', end='12/31/2023')

# 初始化一个空的DataFrame用于存储流量数据
traffic_data = pd.DataFrame(index=date_range)

# 为每天的1-7点设置流量为1000，其他时间段流量为6000
for hour in range(24):
    if hour < 7:
        traffic_data.loc[date_range, hour] = 1000
    else:
        traffic_data.loc[date_range, hour] = 6000

    # 对全时段应用10%的随机波动
for hour in range(24):
    traffic_data.loc[date_range, hour] = traffic_data.loc[date_range, hour] * (1 + np.random.rand(len(date_range)) / 10)
    traffic_data.loc[date_range, hour] = traffic_data.loc[date_range, hour].round(2)  # 四舍五入保留两位小数


traffic_data.to_csv('history_data.csv', index=False)
