#!/bin/bash

# 加载模块
module load anaconda/2020.11
module load cuda/10.0
module load cudnn/7.6.5.32_cuda10.0
module load gcc/7.3

source activate openmmlab_det

# 刷新日志缓存
export PYTHONUNBUFFERED=1

# 视频处理
python /data/home/scv9243/run/mmdetection/work/video_ballon.py