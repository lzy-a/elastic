import matplotlib.pyplot as plt
import numpy as np


# 自定义样本点
auc = np.linspace(0, 0.0035, 10)  # 从0到0.0035均匀取10个点
ctr = np.array([0, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 1.9, 2.0, 2.1])  # 自定义CTR数据点
pv = np.array([0, 0.2, 0.4, 0.6, 0.9, 1.2, 1.4, 1.6, 1.65, 1.7])  # 自定义PV数据点
gmv = np.array([0, 0.1, 0.2, 0.5, 1.2, 1.5, 1.8, 2.0, 1.9, 2.0])
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置全局字体为中文宋体

# 绘制折线图（无数据点）并使用浅颜色
# plt.plot(auc, ctr, label="CTR", linestyle='-', color='lightblue', linewidth=2.0)
# plt.plot(auc, pv, label="PV", linestyle='-', color='lightcoral', linewidth=2.0)
# plt.plot(auc, gmv, label="GMV", linestyle='-', color='burlywood', linewidth=2.0)
# plt.xlabel('AUC 提升(%)')
# plt.ylabel('CTR/PV/GMV(%)')


plt.grid(True)
plt.legend()
plt.show()

