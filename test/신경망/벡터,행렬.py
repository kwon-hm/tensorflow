# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 11:59:22 2018

@author: Administrator
"""

import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))



x=np.array([11,12])


신경망={}
신경망['w1']=np.random.rand(2,2)
신경망['w2']=np.random.rand(2,2)

def 예측하다(신경망, x):
    f1합 = np.dot(x, 신경망['w1'])#(2차)벡터
    f1출력 = sigmoid(f1합)#(2차)벡터

    f2합 = np.dot(f1출력, 신경망['w2'])
    f2출력 = sigmoid(f2합)
    return f2출력