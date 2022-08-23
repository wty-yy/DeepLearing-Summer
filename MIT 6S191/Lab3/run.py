# -*- coding: UTF-8 -*-

"""
@author: wty-yy
@software: PyCharm
@file: run.py
@time: 2022/8/18 11:00
"""

from cartpole import Cartpole

for i in range(0, 10):
    cartpole = Cartpole(load=False, output_model='my_dqn' + str(i))
    cartpole.train(80)