# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:19:44 2018

@author: Administrator
"""
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))
    
def 학습하다(x,신경망, t):
    
    o1 = sigmoid(np.dot(x,신경망['w1'])+신경망['b1'])#순전파
    o2 = sigmoid(np.dot(o1,신경망['w2'])+신경망['b2'])  #역전파
   
    #2. 가중치 갱신
    #2.1 에러구하기
    e2 = t - o2
    e1 = np.dot(신경망['w2'],e2)
    #2.2 가중치에 대한 변화율
    #print('shape:',np.shape(e2*o2*(1-o2)))
              #(50*1)              *               (1*10)
    dE_dw2 = np.array(e2, ndmin=2) * np.array(o1,ndmin=2).T#(50*10)
    dE_db2 = np.array(e2*o2*(1-o2))
    dE_dw1 = np.array(e1*o1*(1-o1), ndmin=2) * np.array(x,ndmin=2).T#(25*50)
    dE_db1 = np.array(e1*o1*(1-o1))
    
    #2.3 가중치 갱신 계산
    신경망['w2'] += 0.05*dE_dw2
    신경망['b2'] += 0.05*dE_db2
    신경망['w1'] += 0.05*dE_dw1
    신경망['b1'] += 0.05*dE_db1
    
    global 총에러
    E.append(np.sum(np.square(e2)))
    print('총에러:', E[-1])


def 예측하다(x, 신경망):
    _1층합 = np.dot(x,신경망['w1'])+신경망['b1']
    o1 = sigmoid(_1층합)
    o2 = sigmoid(np.dot(o1,신경망['w2'])+신경망['b2'])
    return o2

#=============================
E=[]
신경망={}
입력노드수=25
출력노드수=10    
은닉노드수=50
신경망['w1'] = np.random.randn(입력노드수,은닉노드수)
신경망['b1'] = np.random.randn(은닉노드수)
신경망['w2'] = np.random.randn(은닉노드수,출력노드수)
신경망['b2'] = np.random.randn(신경망['w2'].shape[1])


#=============================

#1. 학습데이터 준비
xs = np.array(  [[1,1,1,1,1,
                  1,0,0,0,1,
                  1,0,0,0,1,
                  1,0,0,0,1,
                  1,1,1,1,1],
                 [0,0,1,0,0,
                  0,0,1,0,0,
                  0,0,1,0,0,
                  0,0,1,0,0,
                  0,0,1,0,0],
                 [1,1,1,1,1,
                  0,0,0,0,1,
                  1,1,1,1,1,
                  1,0,0,0,0,
                  1,1,1,1,1],
                 [1,1,1,1,1,
                  0,0,0,0,1,
                  1,1,1,1,1,
                  0,0,0,0,1,
                  1,1,1,1,1],
                 [1,0,0,0,1,
                  1,0,0,0,1,
                  1,1,1,1,1,
                  0,0,0,0,1,
                  0,0,0,0,1],
                 [1,1,1,1,1,
                  1,0,0,0,0,
                  1,1,1,1,1,
                  0,0,0,0,1,
                  1,1,1,1,1],
                 [1,0,0,0,0,
                  1,0,0,0,0,
                  1,1,1,1,1,
                  1,0,0,0,1,
                  1,1,1,1,1],
                 [1,1,1,1,1,
                  0,0,0,0,1,
                  0,0,0,0,1,
                  0,0,0,0,1,
                  0,0,0,0,1],
                 [1,1,1,1,1,
                  1,0,0,0,1,
                  1,1,1,1,1,
                  1,0,0,0,1,
                  1,1,1,1,1],
                 [1,1,1,1,1,
                  1,0,0,0,1,
                  1,1,1,1,1,
                  0,0,0,0,1,
                  0,0,0,0,1]])
    
ts =np.array(   [[1,0,0,0,0,0,0,0,0,0],
                 [0,1,0,0,0,0,0,0,0,0],
                 [0,0,1,0,0,0,0,0,0,0],
                 [0,0,0,1,0,0,0,0,0,0],
                 [0,0,0,0,1,0,0,0,0,0],
                 [0,0,0,0,0,1,0,0,0,0],
                 [0,0,0,0,0,0,1,0,0,0],
                 [0,0,0,0,0,0,0,1,0,0],
                 [0,0,0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,0,0,1]])
#2. 학습
for i in range(1000):
    for x, t in zip(xs, ts):
        학습하다(x,신경망,t)
#=====================================
        
# 예측
print('========예측1=============') 
for x, t in zip(xs, ts):
    print('예측:',np.argmax(예측하다(x,신경망)), '실제:',np.argmax(t))
print('========예측2=============')   
for index in np.random.choice(10, 10):
    print('예측:', np.argmax(예측하다(xs[index],신경망)), '실제:', np.argmax(ts[index]))
    
from matplotlib import pyplot as plt

xs_ = [x for x in range(len(E))]
ys_ = [E[x] for x in xs_]

plt.rcParams["figure.figsize"] = (10,5)
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.grid'] = True 
plt.plot(xs_, ys_, 'r')
plt.title('Error Cost')
plt.ylabel('cost' )
plt.xlabel('training time' )
plt.show()
    