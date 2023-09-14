#!/bin/bash
#等待service创建
sleep 5

# 获取endpoint_ip
endpoint_ip=$(python -c "import socket; print(socket.gethostbyname('elastic-master-service.default.svc.cluster.local'))")

#if [ "$endpoint_ip" == "$POD_IP" ]; then
#  # 执行Python脚本
#  python modify.py
#fi

# 执行torchrun命令
torchrun \
    --nnodes=1:3 \
    --nproc_per_node=1 \
    --max_restarts=3 \
    --rdzv_id=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$endpoint_ip:1234" \
    train_elastic.py
