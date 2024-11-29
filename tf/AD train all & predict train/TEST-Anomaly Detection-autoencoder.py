import os
import math
from matplotlib import colors
import numpy as np
import pandas as pd
import seaborn as sns
from time import sleep
import mongoDB as mongo
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
plt.style.use('fivethirtyeight')
conn = mongo.conn('deviceStats_1m')

mem, cpu, network, time, day = [], [], [], [], []

print('==========================')
print('==========================')

TIME_STEPS = 10
rate = 0.001
shuffle = False
batch_size = 32
validation_split = 0.1
epochs = 100
THRESHOLD = 0.05
path = 'AD train all & predict train/'
name = 'checkpoint(e-50, b-32, t-10, r-0.1, v-0.1)'
print('TIME_STEPS: ', TIME_STEPS)
print('rate: ', rate)
print('shuffle: ', shuffle)
print('batch_size: ', batch_size)
print('validation_split: ', validation_split)
print('epochs: ', epochs)
print('THRESHOLD: ', THRESHOLD)
checkpoint_path = path + "checkpoint/cp.ckpt"
print('checkpoint_path: ', checkpoint_path)

print('==========================')
print('==========================')

# 모델의 가중치를 저장하는 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

i_list = conn.find({'dynamicStat': 1}).sort('_id', 1)
for i in i_list:
    date = datetime.fromtimestamp(i['time']['current'] // 1000)
    time.append(date)

    cpu_used = i['currentLoad']['currentload']
    cpu.append(cpu_used)

    mem_used = i['mem']['used']/ 1024 / 1024 / 1024
    mem.append(mem_used)

    if(i['mem']['used'] > 29081968128.0):
        print('!!!!!!!!!!!!!!!!!!', i['time']['current']//1000)

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
# plt.title('Model')
# plt.xlabel('Date')
# plt.ylabel('Used')
# plt.plot(df[['Cpu', 'Mem', 'Network']])
# plt.legend(['Cpu', 'Mem', 'Network'], loc = 'lower right')
# plt.show()

"""
2. Model data
"""
training_data_len = int(len(df))
test_size = len(df) - training_data_len

scaler = MinMaxScaler()
scaler.fit(df)
df[['Cpu', 'Mem', 'Network']] = scaler.transform(df[['Cpu', 'Mem', 'Network']])

df_train = df[0:training_data_len]

def create_dataset(X, time_steps):
    xs, ys = [], []
    for i in range(len(X) - time_steps):
        xs.append(X[i: (i + time_steps)].values.reshape(time_steps,3)),
        ys.append(X[i + 1: (i + time_steps) + 1].values.reshape(time_steps,3))
    return np.array(xs), np.array(ys)

x_train, y_train = create_dataset(
    df_train[['Cpu', 'Mem', 'Network']], TIME_STEPS
)
print('--------------------------')
print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)
print('--------------------------')

"""
3. Model & learning
"""
def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(keras.layers.Dropout(rate=rate))
    model.add(keras.layers.RepeatVector(TIME_STEPS))
    model.add(keras.layers.LSTM(64, activation='relu', return_sequences=True))
    model.add(keras.layers.Dropout(rate=rate))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(x_train.shape[2])))
    # model.add(keras.layers.Dense(32))
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    return model

if(os.path.isfile(path + name + '/saved_model.pb')):
    model = keras.models.load_model(path + name)
else:
    model = create_model()
    history = model.fit(
        x_train,
        y_train,
        shuffle=shuffle,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[cp_callback],
        epochs=epochs
    )
    # loss, acc = model.evaluate(x_train, y_train, verbose=2)
    # print("정확도: {:5.2f}%".format(100*acc))
    model.save(path + 'checkpoint')
    print('loss: ',history.history['loss']);print()
    plt.figure(figsize = (10,5))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.show()

"""
4. Reconstruction error
"""
X_train_pred = model.predict(x_train)
# print('1: ', np.square(X_train_pred - x_train)[0])
# print('2: ', np.mean(np.square(X_train_pred - x_train), axis=1)[0])
# print('3: ', np.max(np.mean(np.square(X_train_pred - x_train), axis=1), axis=1)[0])
train_mae_loss = np.max(np.mean(np.square(X_train_pred - x_train), axis=1), axis=1)
train_mae_loss_min = np.min(np.mean(np.square(X_train_pred - x_train), axis=1), axis=1)
sns.distplot(train_mae_loss, bins=100, kde=True)
plt.show()

"""
5. Threshold
"""
# X_test_pred = model.predict(x_test)
# test_mae_loss = np.max(np.mean(np.square(X_test_pred - x_test), axis=1), axis=1)
# print('--------------------------')
# print('test_mae_loss: ', test_mae_loss)
# print('--------------------------')

# test_score_df = pd.DataFrame(index=df_test[:-(TIME_STEPS)].index)
# test_score_df['loss'] = test_mae_loss
# test_score_df['threshold'] = THRESHOLD
# test_score_df['anomaly'] = test_mae_loss > test_score_df.threshold
# test_score_df['memory'] = df_test[:-(TIME_STEPS)].Mem
# test_score_df['cpu'] = df_test[:-(TIME_STEPS)].Cpu

# print('--------------------------')
# print('test_score_df: ', test_score_df)
# print('--------------------------')
# plt.plot(test_score_df.index, test_score_df.loss, label='loss')
# plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
# # plt.plot(test_score_df.index, test_score_df.ts, label='ts')
# plt.xticks(rotation=25)
# plt.show()

# anomalies = test_score_df[test_score_df.anomaly == True]
# anomalies.describe()
# print('--------------------------')
# print('anomalies.describe(): ', anomalies.describe())
# print('--------------------------')
# print('anomalies: ', anomalies)
# print('--------------------------')

"""
6. Anomaly point
"""
# plt.plot(df_test[:][['Cpu', 'Mem', 'Network']])
# plt.plot(
#     anomalies.index,
#     anomalies.cpu,
#     'o',
#     color='green',
#     label='anomaly'
# )
# plt.plot(
#     anomalies.index,
#     anomalies.memory,
#     'o',
#     color='orange',
#     label='anomaly'
# )
# plt.xticks(rotation=25)
# plt.show()

def create_test_dataset(time_data, cpu_data, mem_data, network_data, time_steps_data):
    df_ = pd.DataFrame()
    df_['Date'] = time_data
    df_['Cpu'] = cpu_data
    df_['Mem'] = mem_data
    df_['Network'] = network_data
    df_.set_index('Date', inplace=True) 
    df_ = df_.loc[:,['Cpu', 'Mem', 'Network']] 
    df_ = df_.astype(float)
    df_test_ = df_[:]
    # df_test_[['Cpu', 'Mem', 'Network']] = scaler.transform(df_test_[['Cpu', 'Mem', 'Network']])
    xx = []
    xx.append(df_test_[:].values.reshape(time_steps_data,3))
    return np.array(xx)

def create_new_test_dataset(time_, cpu_, mem_, network_, time_steps_data, test_predic):
    df_ = pd.DataFrame()
    time_.append(time_[-1] + timedelta(seconds=1))
    cpu_.append(test_predic[-1][0])
    mem_.append(test_predic[-1][1])
    network_.append(test_predic[-1][2])
    df_['Date'] = time_
    df_['Cpu'] = cpu_
    df_['Mem'] = mem_
    df_['Network'] = network_
    df_.set_index('Date', inplace=True) 
    df_ = df_.loc[:,['Cpu', 'Mem', 'Network']] 
    df_ = df_.astype(float)
    df_test_ = df_[:]
    df_test_[['Cpu', 'Mem', 'Network']] = scaler.transform(df_test_[['Cpu', 'Mem', 'Network']])
    xx = []
    xx.append(df_test_[:].values.reshape(time_steps_data,3))
    return np.array(xx)

mem_, cpu_, network_, time_ = [], [], [], []
print('len(df): ', len(df))
# print('len(df_test): ', df_test)
print(len(df) - training_data_len)

_df_pre = pd.DataFrame()
_df_old = pd.DataFrame()
test_predic_t, test_predic_c, test_predic_m, test_predic_n = [], [], [], []
df_old_t, df_old_c, df_old_m, df_old_n = [], [], [], []

num = 0
while num < 1000 -1:
    db_data = df[num : num+1]
    df_Cpu = df[num : num+1]['Cpu']
    df_Mem = df[num : num+1]['Mem']
    df_Network = df[num : num+1]['Network']
    df_Date = df.index[num : num+1]

    time_.append(df_Date)
    cpu_.append(df_Cpu)
    mem_.append(df_Mem)
    network_.append(df_Network)
    if(len(time_) == TIME_STEPS):
        x_test_ = create_test_dataset(time_, cpu_, mem_, network_, TIME_STEPS)
        tp = model.predict(x_test_)
        df_data = scaler.inverse_transform(df[num+1 : num+2]) # 실제
        predic = scaler.inverse_transform(tp[0]) # 예측
        print(num, " 실제 : ", df_data[0][0], df_data[0][1], df_data[0][2], df.index[num+1])
        print(num, " 예측 : ", predic[-1][0], predic[-1][1], predic[-1][2])
        print()
        test_predic_t.append(df.index[num+1])
        test_predic_c.append(predic[0][0])
        test_predic_m.append(predic[0][1])
        test_predic_n.append(predic[0][2])

        df_old_t.append(df.index[num+1])
        df_old_c.append(df_data[0][0])
        df_old_m.append(df_data[0][1])
        df_old_n.append(df_data[0][2])

        time_.pop(0)
        cpu_.pop(0)
        mem_.pop(0)
        network_.pop(0)
    num += 1

_df_pre['Date'] = test_predic_t
_df_pre['Cpu'] = test_predic_c
_df_pre['Mem'] = test_predic_m
_df_pre['Network'] = test_predic_n
_df_pre.set_index('Date', inplace=True) 
_df_pre = _df_pre.loc[:,['Cpu', 'Mem', 'Network']] 
_df_pre = _df_pre.astype(float)

_df_old['Date'] = df_old_t
_df_old['Cpu'] = df_old_c
_df_old['Mem'] = df_old_m
_df_old['Network'] = df_old_n
_df_old.set_index('Date', inplace=True) 
_df_old = _df_old.loc[:,['Cpu', 'Mem', 'Network']] 
_df_old = _df_old.astype(float)
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Used')
plt.plot(_df_old[10:]['Cpu'])
plt.plot(_df_pre[:]['Cpu'], color='red')
plt.legend(['old', 'pre', ], loc = 'lower right')
plt.show()

print('end.')