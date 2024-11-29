# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 08:58:23 2018

@author: Administrator
"""

from matplotlib import pyplot as plt

def f(x):
    return x

xs=[x for x in range(1,100)]
ys=[f(x) for x in xs]

plt.plot(xs,ys, color='green', linestyle='solid')
plt.axis('equal')
plt.show()