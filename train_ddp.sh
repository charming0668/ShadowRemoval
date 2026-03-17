#!/bin/bash
# DDP多卡训练启动脚本

# 设置环境变量
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4

# 获取GPU数量
NGPU=${1:-8}
PORT=${2:-29500}

echo "启动 $NGPU 卡DDP训练..."
torchrun --nproc_per_node=$NGPU --master_port=$PORT shadow_removal/train.py
