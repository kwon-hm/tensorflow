import os
import sys
import math      
import numpy as np
import pandas as pd 
import seaborn as sns
import mongoDB as mongo
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from matplotlib.dates import (MONTHLY, DAILY, HOURLY, MINUTELY, DateFormatter, rrulewrapper, RRuleLocator, drange)

conn = mongo.conn("deviceStats")
print('==========================')
print('==========================')
TIME_STEPS = 10
dropout = 0.1
units = 64
epochs = 10
batch_size = 128
validation_split = 0.1
shuffle = False
path = 'cpu-autoencoder/'
name = 'checkpoint(e-10, b-32, t-10)'
THRESHOLD = 0
print('TIME_STEPS: ', TIME_STEPS)
print('rate: ', dropout)
print('shuffle: ', shuffle)
print('batch_size: ', batch_size)
print('validation_split: ', validation_split)
print('epochs: ', epochs)
print('THRESHOLD: ', THRESHOLD)
print('==========================')
print('==========================')

mem = []
cpu = []
time = []
day = []

i_list = conn.find({"dynamicStat": 1})
for i in i_list:
  """
  Memory
  """
  mem_used = round(i['mem']['used'] / i['mem']['total'] * 100, 2)
  #print('Memory: ', mem_used)
  mem.append(mem_used)
  """
  Cpu
  """
  cpu_used = round(i['currentLoad']['currentload'], 6)
  #print('Cpu: ', cpu_used)
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
  #print(date)
  time.append(date)
  day.append(date)
"""
1. Data graph
"""
# df = df.rename({'Henry Hub Natural Gas Spot Price, Daily (Dollars per Million Btu)': 'price'}, axis = 'columns')
# df = df.reset_index()
df = pd.DataFrame(cpu)
df['Cpu'] = cpu
df['Date']= time
df.set_index('Date', inplace=True) 
df = df.loc[:,['Cpu']] 
df = df.astype(float)
print(df)
print(df.shape)
plt.plot(df, label='Cpu price')
plt.show()

"""
2. Make train, test data
"""
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(train.shape, test.shape)

scaler = StandardScaler()
scaler = scaler.fit(train[['Cpu']])
# robust = RobustScaler(quantile_range=(25, 75)).fit(train[['Cpu']])
train['Cpu'] = scaler.transform(train[['Cpu']])
test['Cpu'] = scaler.transform(test[['Cpu']])

# helper function
def create_dataset(x, y, time_steps=1):
  a, b = [], []
  for i in range(len(x) - time_steps):
    v = x.iloc[i:(i + time_steps)].values
    a.append(v)
    b.append(y.iloc[i + time_steps])
  return np.array(a), np.array(b)

# We’ll create sequences with 30 days of historical data

# reshape to 3D [n_samples, n_steps, n_features]

X_train, y_train = create_dataset(train[['Cpu']], train.Cpu, TIME_STEPS)
X_test, y_test = create_dataset(test[['Cpu']], test.Cpu, TIME_STEPS)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

"""
3. Make train model
"""
# define model 

def create_model():
  model = keras.models.Sequential()
  model.add(keras.layers.LSTM(units=units, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
  model.add(keras.layers.Dropout(rate=dropout))
  model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
  model.add(keras.layers.LSTM(units=units, activation='relu', return_sequences=True))
  model.add(keras.layers.Dropout(rate=dropout))
  # model.add(keras.layers.Dense(25))
  model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
  model.compile(optimizer='adam', loss='mae')
  model.summary()
  return model
# model.summary()
# fit model
"""
4. Training
"""
if(os.path.isfile(path + name + '/saved_model.pb')):
    model = keras.models.load_model(path + name)
else:
  model = create_model()
  history = model.fit(
              X_train, 
              y_train, 
              epochs=epochs, 
              batch_size=batch_size,
              validation_split=validation_split,
              shuffle=shuffle
            )
  model.save(path + 'checkpoint')
  print('===loss===',history.history['loss'])
  print('===val_loss===',history.history['val_loss'])

  """
  5. Train, test loss graph
  """
  plt.figure(figsize = (10,5))
  plt.plot( history.history['loss'], label='train')
  plt.plot( history.history['val_loss'], label='validation')
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoqch')
  plt.legend(['train', 'test'], loc='upper left')
  # plt.yticks(np.arange(0, 1, 0.1))
  # rule = rrulewrapper(HOURLY)
  plt.show()

"""
6. Loss graph - history for loss
"""
train_pred = model.predict(X_train)
train_loss = np.mean(np.abs(train_pred - X_train), axis=1)
sns.distplot(train_loss, bins=50, kde=True)
avg_loss = train_loss.mean()
print()
print('Training train_loss: ', train_loss)
print('Training avg_loss: ', avg_loss)
print()

# plt.figure(figsize = (10,5))
# sns.histplot(train_loss, bins=50, kde=True)
# plt.show()

"""
7. Get RMSE, threshold
"""
# MAE on the test data:
y_pred = model.predict(X_test)
print('Predict shape:', y_pred.shape); print()
mae = np.mean(np.abs(y_pred - X_test), axis=1)
# reshaping prediction
pred = y_pred.reshape((y_pred.shape[0] * y_pred.shape[1]), y_pred.shape[2])
print('Prediction:', pred.shape); print()
print('Test data shape:', X_test.shape); print()
# reshaping test data
X_test = X_test.reshape((X_test.shape[0] * X_test.shape[1]), X_test.shape[2])
print('Test data:', X_test.shape); print()
# error computation
errors = X_test - pred
print('Error:', errors.shape); print()
# rmse on test data
RMSE = math.sqrt(mean_squared_error(X_test, pred))
print('Test RMSE: %.3f' % RMSE); print()

"""
8. Test & Predicted data
"""
dist = np.linalg.norm(X_test - pred, axis=1)
scores = dist.copy()
print('Score:', scores.shape)
scores.sort()
cut_off = int(0.90 * len(scores))
print('Cutoff value:', cut_off)
threshold = scores[cut_off]
print('Threshold value: ', threshold)
print()

plt.figure(figsize=(14,5))
plt.plot(X_test, color = 'green')
plt.plot(pred, color = 'red')
plt.title('Test & Predicted data')
plt.show()

"""
9. Threshold, loss graph
"""
score = pd.DataFrame(index=test[TIME_STEPS:].index)
score['loss'] = mae
score['threshold'] = threshold
score['anomaly'] = score['loss'] > score['threshold']
score['Cpu'] = test[TIME_STEPS:].Cpu

plt.figure(figsize = (10,5))
plt.plot(score.index, score['loss'], color = 'green', label = 'loss')
plt.plot(score.index, score['threshold'], color = 'r', label = 'threshold')
plt.xticks(rotation = 90)
plt.show()

"""
10. Anomalies
"""
anomalies = score[score['anomaly'] == True]
x = pd.DataFrame(anomalies.Cpu)
x = pd.DataFrame(scaler.inverse_transform(x))
x.index = anomalies.index
x.rename(columns = {0: 'inverse_cpu'}, inplace = True)
anomalies = anomalies.join(x, how = 'left')
anomalies = anomalies.drop(columns=['Cpu'], axis=1)
anomalies.tail(10)
print(anomalies)

"""
11. Test
"""
test_inv = pd.DataFrame(scaler.inverse_transform(test[TIME_STEPS:]))
print(test_inv)
test_inv.index = test[TIME_STEPS:].index
print(test_inv.index)
test_inv.rename(columns = {0: 'Cpu'}, inplace = True)
print(test_inv)

plt.figure(figsize = (10,5))
plt.plot(test_inv.index, test_inv.Cpu, color = 'gray', label='spot cpu')
# sns.scatterplot((anomalies.index, anomalies['inverse_cpu']), color=sns.color_palette()[3], s=55, label='anomaly')
plt.xticks(rotation = 90)
plt.legend(loc='upper center')
plt.show()

num = 0
time_ = []
cpu_ = []
while num < TIME_STEPS*2:
  db_data = train[num : num+1]
  df_Cpu = train[num : num+1]['Cpu']
  time_.append(db_data)
  cpu_.append(df_Cpu)
  if(len(time_) == TIME_STEPS):
    a = []
    df_ = pd.DataFrame()
    df_['Cpu'] = cpu_
    df_['Date']= time_
    df_.set_index('Date', inplace=True) 
    df_ = df_.loc[:,['Cpu']] 
    df_ = df_.astype(float)
    v = df_[['Cpu']].iloc[:].values
    a.append(v)
    train_test = np.array(a)
    tt = model.predict(train_test)
    df_data = scaler.inverse_transform(train[num+1 : num+2]) # 실제
    predic = scaler.inverse_transform(tt) # 예측
    print(num, " 실제 : ", df_data[0][0], train.index[num+1])
    print(num, " 예측 : ", predic[0][0])
    print();print()
    time_.pop(0)
    cpu_.pop(0)
  num += 1

