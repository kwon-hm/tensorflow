# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:15:16 2018

@author: Administrator
"""

from matplotlib import pyplot as plt

def f1(x):
    return x

def f2(x):
    return 2*x

def f3(x):
    return x/2

xs=[x for x in range(1,100)]
ys1=[f1(x) for x in xs]
ys2=[f2(x) for x in xs]
ys3=[f3(x) for x in xs]

plt.plot(xs,ys1, color='green', linestyle='solid', label='-any line1')
plt.plot(xs,ys2, 'r-.',label='-any line2')
plt.plot(xs,ys3,'b:', label='-any curve')
plt.axis('equal')
plt.legend(loc=1)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('nice graph')
plt.show()