#!/bin/bash

# 加载模块
module load anaconda/2020.11
module load cuda/10.0
module load cudnn/7.6.5.32_cuda10.0
module load gcc/7.3

source activate openmmlab_det

# 刷新日志缓存
export PYTHONUNBUFFERED=1

# 训练模型
python tools/train.py configs/mask_rcnn_ballon/mask_rcnn_ballon_101.py --work-dir work/mask_rcnn_ballon_101