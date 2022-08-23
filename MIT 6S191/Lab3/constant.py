# -*- coding: UTF-8 -*-

"""
@author: wty-yy
@software: PyCharm
@file: constant.py
@time: 2022/8/18 10:59
"""
import pathlib
# 该包用于存储与算法有关的常量
# Main Program
ENV_NAME = 'CartPole-v1'
TOTAL_EPISODE = 60  # 如果没有设定游戏次数，该值为默认次数
MAX_STEP = 500  # 游戏环境上限500步

# DQN Algorithm
LEARNING_RATE = 1e-3
MEMORY_SIZE = 10000
EPSILON_INIT = 0.1
EPSILON_MAX = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.001
TERMINAL_DATE_RATE = 0.3  # 设置终止数据在训练数据中的占比
EPOCHS_MIN = 1
EPOCHS_INCREASE = 2
EPOCHS_MAX = 128

BATCH_SIZE = 32
GAMMA = 0.95

MODEL_SAVE_DIR = pathlib.Path('./training_checkpoints')
