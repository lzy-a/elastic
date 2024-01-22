import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt

# 您提供的数据（示例数据，请替换为您的实际数据）
batch_sizes = np.array([64, 128, 256, 512, 1024, 4096, 8192, 16384, 32768, 65536])
data_time = np.array([0.0004872754626803928, 0.0007830895317925348, 0.001452011295273916, 0.002707496907638474, 0.005559433696977532, 0.025432062149047852, 0.052063547240363224, 0.10653213395012749, 0.18428711537961606, 0.2887822124693129])
forward_time = np.array([0.001521140988667806, 0.0015185254838731555, 0.001536131328329534, 0.0015552901074300822, 0.0015602376725938586, 0.0016234119733174641, 0.001731962627834744, 0.001873620351155599, 0.0019423520123517072, 0.001955827077229818])
loss_time = np.array([0.00265152710808648, 0.0026409353468153213, 0.0026286570027994273, 0.0026387974292946806, 0.0026626207992832703, 0.005249226093292236, 0.005476519796583388, 0.005729998482598198, 0.006769860232317889, 0.01048466894361708])
optimizer_time = np.array([0.002744463751051161, 0.0027477397070990667, 0.002752767118698407, 0.0027518242028108826, 0.0027536842893447554, 0.002768852975633409, 0.002795320087009006, 0.006654744678073459, 0.012456611350730614, 0.01819234424167209])
step_time = np.array([0.007473639043172201, 0.007761578326755099, 0.008445500988972116, 0.009736940359613698, 0.012620897400395948, 0.035166088740030924, 0.06216595967610677, 0.1209075927734375, 0.2055911929519088, 0.31954730881585014])

# 构建矩阵 A
A = np.vstack([batch_sizes, np.ones_like(batch_sizes)]).T

# 使用 NNLS 求解
x_data_time, _ = nnls(A, data_time)
x_forward_time, _ = nnls(A, forward_time)
x_loss_time, _ = nnls(A, loss_time)
x_optimizer_time, _ = nnls(A, optimizer_time)
x_step_time, _ = nnls(A, step_time)

# 生成拟合曲线
fit_data_time = np.dot(A, x_data_time)
fit_forward_time = np.dot(A, x_forward_time)
fit_loss_time = np.dot(A, x_loss_time)
fit_optimizer_time = np.dot(A, x_optimizer_time)
fit_step_time = np.dot(A, x_step_time)

# 绘制拟合曲线
plt.figure(figsize=(10, 6))

plt.subplot(2, 3, 1)
plt.scatter(batch_sizes, data_time, label='Actual Data')
plt.plot(batch_sizes, fit_data_time, label='Fitted Curve', color='red')
plt.title('Data Time')

plt.subplot(2, 3, 2)
plt.scatter(batch_sizes, forward_time, label='Actual Data')
plt.plot(batch_sizes, fit_forward_time, label='Fitted Curve', color='red')
plt.title('Forward Time')

plt.subplot(2, 3, 3)
plt.scatter(batch_sizes, loss_time, label='Actual Data')
plt.plot(batch_sizes, fit_loss_time, label='Fitted Curve', color='red')
plt.title('Loss Time')

plt.subplot(2, 3, 4)
plt.scatter(batch_sizes, optimizer_time, label='Actual Data')
plt.plot(batch_sizes, fit_optimizer_time, label='Fitted Curve', color='red')
plt.title('Optimizer Time')

plt.subplot(2, 3, 5)
plt.scatter(batch_sizes, step_time, label='Actual Data')
plt.plot(batch_sizes, fit_step_time, label='Fitted Curve', color='red')
plt.title('Step Time')

plt.tight_layout()
plt.show()