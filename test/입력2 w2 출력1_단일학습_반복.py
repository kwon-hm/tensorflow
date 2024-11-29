# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:41:29 2018

@author: Administrator
"""
import numpy as np
#xs=[[1,2],[2,3],[3,4],[4,5],[5,6]]
#ts=[3,5,7,9,11]
xs=[i for i in range(1, 100+1)]
ts=[2*x for x  in xs]

#목적: 학습을 통한 적절한 W값 구하기
#입력2  출력 2x1 
def 예상하다(x1, w, b):
    return (w * x1) +b 

def 에러구하다(x1, w, b, 실제값):
    return 실제값 - 예상하다(x1, w, b)

def 에러비용구하다(x1, w, b, 실제값): #자승의 합에 의한, 에러율 제곱의 
        return np.sum(np.square(에러구하다(x1, w, b, 실제값)))
    
def w들에대한에러변화율을구하다(x1, w, b, 실제값):# 수치미분이용
    h = 1e-4
    w변화에대한에러변화율 = (에러비용구하다(x1, w+h, b, 실제값) - 에러비용구하다(x1, w-h, b, 실제값)) / (2 * h) 
    b변화에대한에러변화율 = (에러비용구하다(x1, w, b+h, 실제값) - 에러비용구하다(x1, w, b-h, 실제값)) / (2 * h) 
    return w변화에대한에러변화율,b변화에대한에러변화율

w에대한에러변화율 = 1
b에대한에러변화율 = 1

def 최소오류율위한w들을구하다(x1, w, b, 실제값):# by 경사하강법
    테스트w = w
    테스트b = b
    global w에대한에러변화율
    global b에대한에러변화율
    for i in range(1000):
        if np.absolute(w에대한에러변화율) < 0.0002 and np.absolute(b에대한에러변화율) < 0.0002:
           break
        w에대한에러변화율, b에대한에러변화율 = w들에대한에러변화율을구하다(x1, 테스트w, 테스트b, 실제값)
        테스트w -= w에대한에러변화율 * 0.00000001
        테스트b -= b에대한에러변화율 * 0.00000001
    print('w에대한에러변화율:', w에대한에러변화율, 'b에대한에러변화율:', b에대한에러변화율)   
    return 테스트w, 테스트b  
 
np.random.seed(1)
w = np.random.random()# 
b = np.random.random()#
#학습

for x, 실제값 in zip(xs,ts):
    w, b = 최소오류율위한w들을구하다(x, w, b, 실제값)

#Test
for x, t in zip(xs,ts):
    y = 예상하다(x, w, b)
    print('예상:',y, '실제:', t)    

print('w:',w, 'b:', b) 
    
    
