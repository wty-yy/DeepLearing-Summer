# -*- coding: UTF-8 -*-

"""
@author: wty-yy
@software: PyCharm
@file: dqn.py
@time: 2022/8/18 17:20
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示TensorFlow的警告

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
from collections import deque
# 导入自定义包
import constant as const

class DQN:
    def __init__(self, state_dim, action_dim, model_name, load=False):
        self.checkpoint_prefix = const.MODEL_SAVE_DIR.joinpath(model_name)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=const.MEMORY_SIZE)
        self.epsilon = const.EPSILON_MAX

        self.model = self.make_model()
        if load:
            self.model.load_weights(self.checkpoint_prefix).expect_partial()

    def make_model(self):
        model = keras.Sequential([
            layers.Dense(units=32, activation='relu', input_shape=(self.state_dim,)),
            layers.Dense(units=32, activation='relu'),
            layers.Dense(units=self.action_dim)
        ])
        model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam(const.LEARNING_RATE)
        )
        return model

    def act(self, state):  # 根据state返回下一步的行动
        if np.random.rand() < self.epsilon:  # 返回随机的行动
            return np.random.choice(self.action_dim)
        state = np.expand_dims(state, 0)
        return np.argmax(self.model.predict(state)[0])

    def sample(self):  # 根据概率分布进行采样
        prob = []  # 存储每个数据取到的概率
        terminal_prob = const.TERMINAL_DATE_RATE
        continue_prob = 1 - terminal_prob
        total_num = len(self.memory)
        terminal_num = 0
        for i in self.memory:
            if i[4]: terminal_num += 1
        continue_num = total_num - terminal_num
        for i in range(total_num):
            if self.memory[i][4]:  # terminal
                prob.append(terminal_prob / terminal_num)
            else:  # continue
                prob.append(continue_prob / continue_num)
        prob /= np.sum(prob)
        idx = np.random.choice(a=total_num, size=const.BATCH_SIZE, p=prob, replace=False)  # 此处取样一定不能有重复
        return [self.memory[i] for i in idx]

    def experience_replay(self):  # 利用记忆数据进行训练
        if len(self.memory) < const.BATCH_SIZE:
            return 0
        batch = self.sample()  # 从记忆数组中根据数据分布进行采样
        state = np.zeros((const.BATCH_SIZE, self.state_dim))
        state_next = np.zeros((const.BATCH_SIZE, self.state_dim))
        action, reward, terminal = [], [], []
        # sars': 存储数据的顺序为当前状态s, 行动a, 奖励r, 下一个状态s', 最后是判断是否终止游戏
        # 这样将batch中的元素分别提取出来，便于一次做预测，而不是单个单个地做，提高效率
        for i in range(const.BATCH_SIZE):
            state[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            state_next[i] = batch[i][3]
            terminal.append(batch[i][4])
        target = self.model.predict(state)
        y_next = self.model.predict(state_next)
        for i in range(const.BATCH_SIZE):
            target[i][action[i]] = reward[i]
            if not terminal[i]:  # 如果非终止状态，target加上后续预测的Q值
                target[i][action[i]] += const.GAMMA * np.max(y_next[i])
        self.model.fit(state, target, batch_size=const.BATCH_SIZE, verbose=0)
        self.epsilon = max(self.epsilon * const.EPSILON_DECAY, const.EPSILON_MIN)
        return np.mean(target)
