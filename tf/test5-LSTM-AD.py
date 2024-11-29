import sys
import math
import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import mongoDB as mongo

conn = mongo.conn("deviceStats")
print("================= Count: ",conn.count())

mem = []
cpu = []
network = []
time = []
day = []

i_list = conn.find({'dynamicStat': 1})
for i in i_list:
  date = datetime.fromtimestamp(i['time']['current'] // 1000)
  time.append(date)
  day.append(date)

  cpu_used = i['currentLoad']['currentload']
  cpu.append(cpu_used)

  mem_used = i['mem']['used']
  mem.append(mem_used)

  net_used = i['networkStats'][0]['rx_sec']
  network.append(net_used)

"""
1. Data graph
"""
# df = df.rename({'Henry Hub Natural Gas Spot Price, Daily (Dollars per Million Btu)': 'price'}, axis = 'columns')
# df = df.reset_index()
df = pd.DataFrame(cpu)
df = pd.DataFrame(mem)
df = pd.DataFrame(network)
df['Date'] = time
df['Cpu'] = cpu
df['Mem'] = mem
df['Network'] = network
df.set_index('Date', inplace=True) 
df = df.loc[:,['Cpu', 'Mem', 'Network']] 
df = df.astype(float)
print(df)
print(df.shape)
dataset = df.values

plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Used')
plt.plot(df[['Cpu', 'Mem', 'Network']])
plt.legend(['Cpu', 'Mem', 'Network'], loc = 'lower right')
plt.show()

training_data_len = math.ceil(len(dataset) * .95)
print('training_data_len: ', training_data_len)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)
print('scaled_data: ', scaled_data)

train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i])
  y_train.append(train_data[i])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], -1))
print(x_train.shape)

model = keras.models.Sequential()
model.add(keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(keras.layers.LSTM(50, return_sequences=False))
model.add(keras.layers.Dense(25))
model.add(keras.layers.Dense(3))
model.compile(optimizer='adam', loss='mean_squared_error')
hitsory = model.fit(
            x_train, 
            y_train, 
            batch_size=1, 
            epochs=10
          )

test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], -1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
RMSE = mean_squared_error(y_test, predictions)**0.5
print('RMSE: ', RMSE)
print('x_test.shape: ', x_test.shape)
print('predictions.shape: ', predictions.shape)

train = df[:training_data_len]
valid = df[training_data_len:]
valid[['Predictions1', 'Predictions2', 'Predictions3']] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Price Used')
plt.plot(train['Cpu'])
plt.plot(valid[['Cpu', 'Mem', 'Network']])
plt.legend(['Cpu', 'Mem', 'Network'], loc = 'lower right')
plt.show()

print('valid: ', valid)

new_list = conn.find({"dynamicStat": 1}).sort("_id", -1).limit(60)
for i in new_list:
  date = datetime.fromtimestamp(i['time']['current'] // 1000)
  time.append(date)
  day.append(date)

  cpu_used = i['currentLoad']['currentload']
  cpu.append(cpu_used)

  mem_used = i['mem']['used']
  mem.append(mem_used)

  net_used = i['networkStats'][0]['rx_sec']
  network.append(net_used)

new_df = pd.DataFrame(cpu)
new_df['Cpu'] = cpu
new_df['Mem'] = mem
new_df['Network'] = network
new_df['Date'] = time
new_df.set_index('Date', inplace=True)
new_df = new_df.loc[:, ['Cpu', 'Mem', 'Network']]
new_df = new_df.astype(float)
print(df)

new_dataset = new_df.values
new_df_scaled = scaler.transform(new_dataset)

X_test = []
X_test.append(new_df_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], -1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print('pred_price: ', pred_price)