import pandas as pd
import numpy as np
import random

# 定义 quadratic_curve 函数
def quadratic_curve(x):
    if x > 600:
        return 3000 + random.randint(-50, 0)
    else:
        return 8.0 / 300 * x ** 2 - 16 * x + 3000 + random.randint(-100, 100)

# 生成日期范围
dates = pd.date_range(start='2023-01-01 00:00:00', end='2023-12-31 23:45:00', freq='15T')

# 计算 x 值
x_values = 60 * dates.hour + dates.minute

# 使用 quadratic_curve 函数生成 throughput
throughput = [quadratic_curve(x) for x in x_values]

# 创建 DataFrame
df = pd.DataFrame({'date': dates, 'throughput': throughput})

# 将日期格式化为指定的格式
df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

# 保存 DataFrame 到 CSV 文件
df.to_csv('throughput_dataset.csv', index=False)