# -*- coding: UTF-8 -*-

"""
@author: wty-yy
@software: PyCharm
@file: run.py
@time: 2022/8/18 11:00
"""

from cartpole import Cartpole

cartpole = Cartpole(load=False, output_model='Mini_batch3')
cartpole.train(100)