# 使用 pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime 作为基础镜像
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 将项目中的 requirements.txt 复制到镜像中
COPY requirements.txt .

# 使用 pip 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 将项目文件复制到镜像中
COPY . .

# 设置入口点为 torchrun 命令，并设置默认参数
ENTRYPOINT ["torchrun", \
    "--nnodes=1:3", \
    "--nproc_per_node=4", \
    "--max_restarts=3", \
    "--rdzv_id=1", \
    "--rdzv_backend=c10d", \
    "--rdzv_endpoint=192.0.0.1:1234"]

# 设置默认命令（要运行的脚本）
CMD ["train_elastic.py"]