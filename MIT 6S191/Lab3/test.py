# -*- coding: UTF-8 -*-

"""
@author: wty-yy
@software: PyCharm
@file: test.py
@time: 2022/8/19 9:36
"""

import pathlib
import constant as const
from cartpole import Cartpole
import numpy as np

cartpole = Cartpole(load=True, input_model='Mini_batch3_best')
# cartpole.test(seed=1)
cartpole.test()
# cartpole.test(seed=3)
# cartpole.test(seed=19260817)
