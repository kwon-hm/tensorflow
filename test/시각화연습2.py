# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:04:37 2018

@author: Administrator
"""

from matplotlib import pyplot as plt
import numpy as np

np.random.seed(0)

f=lambda x: 2*x

xs=[x for x in range(1, 101) if x % 2 ==0]
ys=[f(x)+np.random.uniform(-30,30) for x in xs]

plt.scatter(xs,ys)
plt.axis('equal')
plt.show()


