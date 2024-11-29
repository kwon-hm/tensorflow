# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:41:29 2018

@author: celab
"""
import numpy as np

input_size = 3
#input_size == n , f(x1, x2, ....,xn, w1, w2, ...., wn, b)값은 스칼라
#f(x1, x2, ....,xn, w1, w2, ...., wn, b) = x1 * w1 + x2 * w2  + .... + xn *wn +  b
#입력:- batch size x input_siza 행렬, ws-numpy array (n차)w값들 벡터, b 스칼라
#반환: 예상값 = 벡터(batch size 차)
def 예상하다(xs_batch, ws, b):
    return np.add(np.dot(xs_batch, ws), b)
#입력:xs-numpy array (n차)x값들 벡터 , ws-numpy array (n차)w값들 벡터, b 스칼라, 실제값: 스칼라 
#반환: 예상값 = 벡터(batch size 차)
def 에러구하다(xs_batch, ws, b, 실제값):
    return np.subtract(실제값, 예상하다(xs_batch, ws, b))
#입력:xs-numpy array (n차)x값들 벡터 , ws-numpy array (n차)w값들 벡터, b 스칼라, 실제값: 스칼라 
#반환: 스칼라
def 에러비용구하다(xs_batch, ws, b, 실제값): #자승의 합에 의한, 에러율 제곱의 
    return np.sum(np.square(에러구하다(xs_batch, ws, b, 실제값)))
#입력:xs-numpy array (n차)x값들 벡터 , ws-numpy array (n차)w값들 벡터, b 스칼라, 실제값: 스칼라    
#반환: (w변화에대한에러변화율들:input_size차 벡터 , b변화에대한에러변화율_스칼라)인 튜플    
def w들과b에대한에러변화율을구하다(xs_batch, ws, b, 실제값):# 수치미분이용
    h = 1e-4
    w변화에대한에러변화율들= np.zeros_like(ws)
    for i in range(len(ws)):
        원래w_i값 = ws[i]
        ws[i]= 원래w_i값 + h
        에러비용2 = 에러비용구하다(xs_batch, ws, b, 실제값)
        ws[i]= 원래w_i값 - h
        에러비용1 = 에러비용구하다(xs_batch, ws, b, 실제값)        
        w변화에대한에러변화율들[i] = (에러비용2 - 에러비용1) / (2 * h) 
        ws[i] = 원래w_i값
    b변화에대한에러변화율 = (에러비용구하다(xs_batch, ws, b+h , 실제값) - 에러비용구하다(xs_batch, ws, b-h, 실제값)) / (2 * h) 
    return w변화에대한에러변화율들, b변화에대한에러변화율

w변화에대한에러변화율들 = np.ones(input_size)
b에대한에러변화율 = 1
#입력:x1, x2, w1, w2, b, 실제값 모두 스칼라    
#반환: (테스트w1_스칼라, 테스트w2_스칼라, 테스트b_스칼라)인 튜플    
def 최소오류율위한w들과b를구하다(xs, ws, b, 실제값):# by 경사하강법
    테스트ws = ws.copy()
    테스트b = b
    global w변화에대한에러변화율들
    global b에대한에러변화율
    for i in range(10000):
       if sum(np.absolute(w변화에대한에러변화율들) < 0.01)== len(ws) and np.absolute(b에대한에러변화율) < 0.01 :
           break
       w변화에대한에러변화율들, b에대한에러변화율 = w들과b에대한에러변화율을구하다(xs, 테스트ws, b, 실제값)
       테스트ws -= w변화에대한에러변화율들 * 1e-9
       테스트b -= b에대한에러변화율 * 1e-9
    print('dE/dw 들:', w변화에대한에러변화율들, 'dE/dd:', b에대한에러변화율)   
    return 테스트ws, 테스트b  
#======학습==============
#1.학습 데이터 준비 
batch_size = 10
xs=np.array([[x_i for x_i in range(i, i+input_size)] for i in range(1, 100+1)])
input_size = xs.shape[1]#입력 데이터 size를 구함.
ts=np.array([sum(x) for x  in xs])
#2.가중치, 편향 초기화    
np.random.seed(4)
ws = np.random.randn(input_size) 
b = np.random.random()
#3.학습실행
for i in range(len(xs)//batch_size):
    xs_batch = xs[i*batch_size: i*batch_size+batch_size]
    ts_batch = ts[i*batch_size: i*batch_size+batch_size]
    ws, b = 최소오류율위한w들과b를구하다(xs_batch, ws, b, ts_batch)

#========================
#Test
for xs_i, 실제값 in zip(xs,ts):
    예상 = 예상하다(xs_i, ws, b)
    print('예상:', 예상, '실제:', 실제값)    

print('ws:',ws, 'b:', b) 
