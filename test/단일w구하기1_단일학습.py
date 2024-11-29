# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:41:29 2018

@author: Administrator
"""
import numpy as np
#xs=np.array([1,2,3,4,5])
#ts=np.array([2,4,6,8,10])

x=1
t=2

#f(x)=2x
#목적: 학습을 통한 적절한 W값 구하기

def 예상하다(x, w):
    return w * x

def 에러구하다(x, 실제값, w):
    return 실제값 - 예상하다(x, w)

def 에러비용구하다(x,실제값, w): #에러량 제곱의 
        return np.sum(np.square(에러구하다(x, 실제값, w)))
    
def w에대한에러변화율구하다(x, w, 실제값):# 수치미분이용
    h = 1e-4
    w변화에대한에러변화율 = (에러비용구하다(x, 실제값, w + h) - 에러비용구하다(x, 실제값, w - h)) / (2 * h) 
    return w변화에대한에러변화율

def 최소오류율위한w구하다(x, w, 실제값):# by 경사하강법
    테스트w = w
    for i in range(1000000000):
       w에대한에러변화율 = w에대한에러변화율구하다(x, 테스트w, 실제값)
       if np.absolute(w에대한에러변화율) < 0.0001 :
           break
       테스트w -= w에대한에러변화율 * 0.000000001
       print(i, w에대한에러변화율)
    return 테스트w  

np.random.seed(1)
w = np.random.random()# 최적 w=2
w = 최소오류율위한w구하다(1, w, 2)
print(w)











