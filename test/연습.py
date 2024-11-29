# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 14:43:15 2018

@author: Administrator
"""
import numpy as np

batch_size=10
xs=[i for i in range(1,batch_size*10+1)]
ts=[2*x for x in xs]

def 예상하다(x_batch, w, b):
    return np.add(np.dot(x_batch,w),b)

def 에러구하다(x_batch, w, b, 실제값_batch):
    return np.subtract(실제값_batch, 예상하다(x_batch, w, b))

def 에러비용구하다(x_batch,w,b,실제값_batch):
    return np.sum(np.square(에러구하다(x_batch, w, b, 실제값_batch)))

def w와b에대한에러변화율을구하다(x_batch,w,b,실제값_batch):
    h=1e-4
    w변화에대한에러변화율 = (에러비용구하다(x_batch, w+h,b,실제값_batch) - 에러비용구하다(x_batch, w-h,b,실제값_batch))/(2*h)
    b변화에대한에러변화율 = (에러비용구하다(x_batch, w,b+h,실제값_batch) - 에러비용구하다(x_batch, w,b-h,실제값_batch))/(2*h)
    return w변화에대한에러변화율,b변화에대한에러변화율

w에대한에러변화율 = 1
b에대한에러변화율 = 1

def 최소오류들위한w와b를구하다(x_batch,w,b,실제값_batch):
    테스트w = w
    테스트b = b
    global w에대한에러변화율
    global b에대한에러변화율
    for i in range(10000):
        if np.absolute(w에대한에러변화율) <0.0002 and np.absolute(b에대한에러변화율) <0.0002:
            break
        w에대한에러변화율,b에대한에러변화율 = w와b에대한에러변화율을구하다(x_batch,테스트w,테스트b,실제값_batch)
        테스트w -=w에대한에러변화율 *0.000001
        테스트b -=b에대한에러변화율 *0.000001
    print('w에대한에러변화율:', w에대한에러변화율, 'b에대한에러변화율:', b에대한에러변화율)
    return 테스트w,테스트b

np.random.seed(1)
w=np.random.random()
b=np.random.random()

for x,실제값 in zip(xs,ts):
    y=예상하다(x,w,b)
    print('예상:',y,'실제:',실제값)
    
print('w',w,'b',b)