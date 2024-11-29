import math
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
from mongo import mongoDB as mongo
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from time import sleep
plt.style.use('fivethirtyeight')
conn = mongo.conn("deviceStats")

mem, cpu, network, time, day = [], [], [], [], []

print('--------------------------')
TIME_STEPS = 30
print('TIME_STEPS: ', TIME_STEPS)

rate = 0.001
print('rate: ', rate)

shuffle = False
batch_size = 32
validation_split = 0.1
epochs = 100
print('shuffle: ', shuffle)
print('batch_size: ', batch_size)
print('validation_split: ', validation_split)
print('epochs: ', epochs)

THRESHOLD = 0.015
print('THRESHOLD: ', THRESHOLD)
print('--------------------------')

i_list = conn.find({'dynamicStat': 1})
for i in i_list:
    date = datetime.fromtimestamp(i['time']['current'] // 1000)
    time.append(date)

    cpu_used = i['currentLoad']['currentload']
    cpu.append(cpu_used)

    mem_used = i['mem']['used']/ 1024 / 1024 / 1024
    mem.append(mem_used)

    net_used = i['networkStats'][0]['rx_bytes']
    network.append(net_used)

"""
1. Get mongo data
"""
df = pd.DataFrame()
df['Date'] = time
df['Cpu'] = cpu
df['Mem'] = mem
df['Network'] = network
df.set_index('Date', inplace=True) 
df = df.loc[:,['Cpu', 'Mem', 'Network']] 
df = df.astype(float)
print(df)
print(df.shape)
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Used')
plt.plot(df[['Cpu', 'Mem', 'Network']])
plt.legend(['Cpu', 'Mem', 'Network'], loc = 'lower right')
plt.show()

"""
2. Model data
"""
training_data_len = int(len(df) * 0.8)
test_size = len(df) - training_data_len

scaler = MinMaxScaler()
df_train = df[0:training_data_len]
df_train[['Cpu', 'Mem', 'Network']] = scaler.fit_transform(df_train[['Cpu', 'Mem', 'Network']])
df_test = df[training_data_len:len(df)]
df_test[['Cpu', 'Mem', 'Network']] = scaler.fit_transform(df_test[['Cpu', 'Mem', 'Network']])

def create_dataset(X, time_steps=10):
    xs, ys = [], []
    for i in range(len(X) - time_steps -1):
        xs.append(X[i: (i + time_steps)].values.reshape(time_steps,3)),
        ys.append(X[i + 1: (i + time_steps) + 1].values.reshape(time_steps,3))
    return np.array(xs), np.array(ys)

x_train, y_train = create_dataset(
    df_train[['Cpu', 'Mem', 'Network']], TIME_STEPS
)
x_test, y_test = create_dataset(
    df_test[['Cpu', 'Mem', 'Network']], TIME_STEPS
)
print('--------------------------')
print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)
print('--------------------------')

"""
3. Model & learning
"""
model = keras.models.Sequential()
model.add(keras.layers.LSTM(64, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(keras.layers.Dropout(rate=rate))
model.add(keras.layers.RepeatVector(TIME_STEPS))
model.add(keras.layers.LSTM(64, activation='relu', return_sequences=True))
model.add(keras.layers.Dropout(rate=rate))
# model.add(keras.layers.Dense(32))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(x_train.shape[2])))
# model.add(keras.layers.Dense(3))
model.compile(optimizer='adam', loss='mae')
model.summary()

history = model.fit(
    x_train,
    y_train,
    shuffle=shuffle,
    batch_size=batch_size,
    validation_split=validation_split,
    epochs=epochs
)
print('loss: ',history.history['loss']);print()
plt.figure(figsize = (10,5))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.show()

"""
4. Reconstruction error
"""
X_train_pred = model.predict(x_train)
train_mae_loss = np.max(np.mean(np.square(X_train_pred - x_train), axis=1), axis=1)
sns.distplot(train_mae_loss, bins=100, kde=True)
plt.show()

"""
5. Threshold
"""

X_test_pred = model.predict(x_test)
test_mae_loss = np.max(np.mean(np.square(X_test_pred - x_test), axis=1), axis=1)
print('--------------------------')
print('test_mae_loss: ', test_mae_loss)
print('--------------------------')

test_score_df = pd.DataFrame(index=df_test[:-(TIME_STEPS+1)].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_mae_loss > test_score_df.threshold
test_score_df['memory'] = df_test[:-(TIME_STEPS+1)].Mem
print('--------------------------')
print('test_score_df: ', test_score_df)
print('--------------------------')
plt.plot(test_score_df.index, test_score_df.loss, label='loss')
plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
plt.xticks(rotation=25)
plt.show()

anomalies = test_score_df[test_score_df.anomaly == True]
anomalies.describe()
print('--------------------------')
print('anomalies.describe(): ', anomalies.describe())
print('--------------------------')
print('anomalies: ', anomalies)
print('--------------------------')

"""
6. Anomaly point
"""
df_test[:-TIME_STEPS][['Cpu', 'Mem', 'Network']].plot()
plt.plot(
    anomalies.index,
    anomalies.memory,
    'o',
    color='green',
    label='anomaly'
)
plt.xticks(rotation=25)
plt.show()

# for idx, i in enumerate(x_test):
#     TEST = np.reshape(i, (1, i.shape[0], i.shape[1]))
#     X_test_pred = model.predict(TEST)
#     test_mae_loss = np.max(np.mean(np.square(X_test_pred - TEST), axis=1), axis=1)
#     if(test_mae_loss > THRESHOLD):
#         print('***** Start *****')
#         print(tt.Day[idx + training_data_len])
#         print('test_mae_loss: ', test_mae_loss)
#         print('***** End *****');print()
    # sleep(1)