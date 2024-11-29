import mongoDB as mongo
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.dates import (MONTHLY, DAILY, HOURLY, MINUTELY, DateFormatter, rrulewrapper, RRuleLocator, drange)

import pandas as pd 
import tensorflow as tf
from tensorflow import keras
                              
conn = mongo.conn("deviceStats")
print("================= Count: ",conn.count())

mem = []
cpu = []
time = []

i_list = conn.find()
for i in i_list:
  if(i['dynamicStat'] == 1):
    """
    Memory
    """
    mem_used = round(i['mem']['used'] / i['mem']['total'] * 100, 2)
    print('Memory: ', mem_used)
    mem.append(mem_used)
    """
    Cpu
    """
    cpu_used = round(i['currentLoad']['currentload'] / i['currentLoad']['currentload_idle'], 6)
    print('Cpu: ', cpu_used)
    cpu.append(cpu_used)
    """
    Network
    """
    #print(i['networkStats'][0]['rx_bytes']) 
    #print(i['networkStats'][0]['tx_bytes'])
    """
    Day
    """
    date = datetime.fromtimestamp(i['time']['current'] // 1000)
    print(date)
    time.append(date)

# define input sequence
seq_in = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

# reshape input into [samples, timesteps, features]
n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))

# prepare output sequence
seq_out = seq_in[:, 1:, :]
n_out = n_in - 1

# define model 
model = keras.models.Sequential()
model.add(keras.layers.LSTM(100, activation='relu', input_shape=(n_in, 1)))
model.add(keras.layers.RepeatVector(n_out))
model.add(keras.layers.LSTM(100, activation='relu', return_sequences=True))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))
model.compile(optimizer='adam', loss='mse')

# fit model
history = model.fit(
            seq_in, 
            seq_out, 
            epochs=500, 
            verbose=0
          )
print('===loss===',history.history['loss'])
# predict
yhat = model.predict(seq_in)
print(yhat)

plt.plot(time, mem, label='Memory')
plt.plot(time, cpu, label='Cpu')
# rule = rrulewrapper(HOURLY)
plt.legend(loc=1)
plt.ylim([0, 1])
plt.yticks(np.arange(0, 1, 0.1))

plt.title('Memory used(%)')
plt.xlabel('x-Label(Day)')
plt.ylabel('y-Label(used)')
plt.show()

