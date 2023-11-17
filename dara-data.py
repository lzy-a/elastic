import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建日期范围
date_range = pd.date_range(start='10/15/2023', end='10/18/2023', freq='H')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置全局字体为中文宋体
# 初始化一个全0的序列
data = np.zeros(len(date_range))

# 设置每天的1:00 - 7:00范围内的数值大小在1000左右10%波动
for i in range(len(date_range)):
    if date_range[i].hour >= 0 and date_range[i].hour < 8:
        data[i] = 1000 + np.random.uniform(-200, 200)

# 设置每天的其他时间段数值大小在6000左右10%波动
for i in range(len(date_range)):
    if date_range[i].hour < 0 or date_range[i].hour >= 8:
        data[i] = 6000 + np.random.uniform(-600, 1200)

#把每天0点和8点的数据修正为两侧的均值
for i in range(len(date_range)):
    if date_range[i].hour == 0 or date_range[i].hour == 8:
        # 计算均值并替代0点和8点的数据
        if i == 0:
            data[i] = (data[len(date_range) - 1] + data[i + 1]) / 2
        elif i == len(date_range) - 1:
            data[i] = (data[i - 1] + data[0]) / 2
        else:
            data[i] = (data[i - 1] + data[i + 1]) / 2


# 创建一个pandas Series对象
data_series = pd.Series(data, index=date_range)

# 绘制数据
plt.figure(figsize=(12, 6))
plt.plot(data_series.index, data_series.values, color='navy')
plt.xlabel('日期和时间')
plt.ylabel('流量')
plt.legend()
#横坐标显示日期和小时
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
plt.show()
