import pandas as pd
import numpy as np

# 创建日期范围
date_range = pd.date_range(start='1/1/2023', end='12/31/2023', freq='T')

# 初始化一个全0的序列
data = np.zeros(len(date_range))

# 设置每天的1:00 - 7:00范围内的数值大小在1000左右10%波动
for i in range(len(date_range)):
    if date_range[i].hour >= 1 and date_range[i].hour < 7:
        data[i] = 1000 + np.random.uniform(-100, 100)

# 设置每天的其他时间段数值大小在6000左右10%波动
for i in range(len(date_range)):
    if date_range[i].hour < 1 or date_range[i].hour >= 7:
        data[i] = 6000 + np.random.uniform(-600, 600)

# 设置每天的1000到6000以及6000到1000的流量波动，在一小时内快速上升和下降
for i in range(len(date_range)):
    if date_range[i].minute == 0:
        if date_range[i].hour == 1 or date_range[i].hour == 7:
            data[i:i+60] = np.linspace(data[i], data[i+60], 60)

# 在某些分钟（约占1%）出现30%-50%的流量激增
increase_indices = np.random.choice(len(date_range), size=int(0.01*len(date_range)), replace=False)
for i in increase_indices:
    data[i] *= np.random.uniform(1.3, 1.5)

# 创建一个pandas Series对象
data_series = pd.Series(data, index=date_range)



data_series.to_csv('history_data.csv', index=False)
