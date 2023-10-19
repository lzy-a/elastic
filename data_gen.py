import numpy as np
import pandas as pd

# 定义时间段和对应的流量
time_periods = ['1-7', 'other']
traffic_data = [1000, 6000]

# 定义全天的24小时
hours = np.arange(24)

# 生成基础流量数据
base_traffic = np.zeros_like(hours)
for i, period in enumerate(time_periods):
    if period == '1-7':
        base_traffic[hours < 7] = traffic_data[i]
    else:
        base_traffic[hours >= 7] = traffic_data[i]

    # 生成带有10%波动性的新数据
std_dev = 0.1 * base_traffic  # 计算标准差
traffic_data_new = base_traffic + np.random.normal(0, std_dev)  # 生成新数据

# 将新数据转换为DataFrame并打印
df = pd.DataFrame({'Hour': hours, 'Traffic': traffic_data_new})
print(df)
df.to_csv('history_data.csv', index=False)
