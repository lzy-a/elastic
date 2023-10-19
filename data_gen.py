import numpy as np
import pandas as pd

# 设置时间范围和间隔
start_time = 0  # 开始时间（小时）
end_time = 24  # 结束时间（小时）
time_interval = 1  # 时间间隔（小时）

# 生成时间序列
time = np.arange(start_time, end_time, time_interval)

# 生成凌晨1点-早上7点的波谷数据
valley_data = np.zeros_like(time)
valley_data[(time >= 1) & (time <= 7)] = 1

# 生成其他时间的波峰数据
peak_data = np.ones_like(time)
peak_data[(time < 1) | (time > 7)] = 0

# 将波谷和波峰数据进行合并
data = valley_data + peak_data

# 将数据保存到CSV文件
data_df = pd.DataFrame(data, columns=['Data'])
data_df.to_csv('history_data.csv', index=False)
