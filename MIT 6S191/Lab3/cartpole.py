# -*- coding: UTF-8 -*-

"""
@author: wty-yy
@software: PyCharm
@file: cartpole.py
@time: 2022/8/18 10:22
@update: v1.2: 将网络实时更新，转为完成整个游戏后更新一次，这样便于控制总更新次数，提升训练效率，缺点为不能实时提供决策.
               每次更新次数由epochs控制，从1以2的幂级数递增至1024，可以在前期获得大量的随机状态.
         v1.3: 加入对训练数据集选取概率的划分，可以解决终止数据分布过少的问题(数据偏差)
         v1.4: 1. 将路径类型改为pathlib.Path
               2. 绘制Q值的变化曲线
               无法降低学习率，只能使用随机梯度下降进行优化，不能使用过去最优模型给出目标值
         v1.5: 以一个Mini-Batch作为整体更新一次.
               注意: 一个batch取样时绝不能重复取样，则会导致重复训练，容易过拟合，用target Q网络应该可以解决此问题.
         v1.6: 预测改为整体做一次操作，提高效率.
注：gym的版本为 0.21.0
"""

import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# 导入自定义包
import constant as const
from dqn import DQN

class Cartpole:
    def __init__(self, load=False, input_model='my_dqn', output_model='my_dqn'):
        self.model_name = output_model
        self.checkpoint_prefix = const.MODEL_SAVE_DIR.joinpath(output_model)
        self.best_prefix = const.MODEL_SAVE_DIR.joinpath(output_model + '_best')
        self.score = []  # 记录每次游戏的得分(持续时间)
        self.q_value = []  # 记录Q值变化
        self.env = gym.make(const.ENV_NAME)  # type: gym.Env
        state_dim = self.env.observation_space.shape[0]  # 观察的变量数目
        action_dim = self.env.action_space.n  # 行动的变量数目
        self.dqn = DQN(state_dim, action_dim, input_model, load)  # 创建智能体
        self.best_model = self.dqn.make_model()  # 存储得分最高的模型

    def train(self, total_episode=const.TOTAL_EPISODE):
        epochs = const.EPOCHS_MIN  # 游戏结束后网络的更新次数
        for episode in tqdm(range(total_episode)):
            state = self.env.reset()  # 得到初始状态
            q_memory = []  # 记录当前更新所返回的Q值
            for step in range(const.MAX_STEP):
                # env.render()  # 显示游戏画面
                action = self.dqn.act(state)  # 使用epsilon-贪心算法预测下一步行动
                state_next, reward, terminal, info = self.env.step(action)
                if terminal and step != const.MAX_STEP - 1:
                    reward = -reward  # 如果游戏因错误而终止，该步奖励为-1
                self.dqn.memory.append((state, action, reward, state_next, terminal))
                state = state_next
                if terminal:
                    self.env.close()  # 关闭游戏画面
                    self.score.append(step)
                    if step == max(self.score):  # 存储得分最高的模型的权重
                        self.best_model.set_weights(self.dqn.model.get_weights())
                    break
                q_memory.append(self.dqn.experience_replay())  # 对模型进行训练
            self.q_value.append(np.mean(q_memory))  # 保存当前更新Q值的平均值
            # 保存模型当前权重
            self.dqn.model.save_weights(self.checkpoint_prefix)
            self.best_model.save_weights(self.best_prefix)
            self.save_figure()
        self.save()

    def save_figure(self):  # 保存训练过程中的曲线图
        plt.rcParams['axes.linewidth'] = 1  # 图框宽度
        plt.rcParams['figure.dpi'] = 200  # plt.show显示分辨率
        config = {
            "font.family": 'serif',
            "font.size": 16,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
        }
        plt.figure(figsize=(10, 5))
        plt.rcParams.update(config)
        plt.subplot(1, 2, 1)
        plt.title('score')
        plt.plot(self.score)
        plt.subplot(1, 2, 2)
        plt.title('Q Value')
        plt.plot(self.q_value)
        plt.savefig(self.checkpoint_prefix)
        plt.close()

    def save(self):  # 训练结束后保存相关得分数据
        print_info = f'{self.model_name}: {self.score=}\n{self.dqn.epsilon=}\n'
        print(print_info)
        score_prefix = const.MODEL_SAVE_DIR.joinpath('scores.txt')
        with open(score_prefix, 'a') as file:
            file.write(print_info)

    def test(self, seed=1):
        self.dqn.epsilon = const.EPSILON_MIN
        self.env.seed(seed)
        state = self.env.reset()
        for step in range(const.MAX_STEP):
            self.env.render()
            action = self.dqn.act(state)
            state, reward, terminal, info = self.env.step(action)
            if terminal:
                self.env.close()
                print(f'维持时间：{step}')
                break

    def retrain(self, total_episode=const.TOTAL_EPISODE):
        self.dqn.epsilon = const.EPSILON_MIN
        self.train(total_episode)

if __name__ == '__main__':
    pass
