import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def getData(company, data_source, start, end):
  web_data = web.DataReader(company, data_source=data_source, start=start, end=end)
  return web_data

df = getData('AAPL', 'yahoo','2012-01-01', '2021-03-02')
print(df)
print(df.shape)

# data = df.filter(['Close'])
data = df
dataset = data.values
training_data_len = math.ceil(len(dataset) * .95)
print('training_data_len: ', training_data_len)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print('scaled_data: ', scaled_data.shape)

train_data = scaled_data[0:training_data_len, :]
print('train_data: ', train_data.shape)
x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i])
  y_train.append(train_data[i])
  # if i<=60:
    # print('x_train: ', x_train)
    # print('y_train: ', y_train)
    # print()

x_train, y_train = np.array(x_train), np.array(y_train)
print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)
print('x_train.shape[0]: ', x_train.shape[0])
print('x_train.shape[1]: ', x_train.shape[1])
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], -1))
print('x_train: ', x_train.shape)
print('x_train.shape: ', x_train.shape)

model = keras.models.Sequential()
model.add(keras.layers.LSTM(120, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(keras.layers.LSTM(120, return_sequences=False))
# model.add(Dropout(rate=0.01))
model.add(keras.layers.Dense(60))
# model.add(Dropout(rate=0.01))
model.add(keras.layers.Dense(6))
model.compile(optimizer='adam', loss='mae')
history = model.fit(
  x_train,
  y_train,
  verbose=1,
  batch_size=32,
  epochs=10
)

print('loss: ', history.history['loss'])

test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i])

x_test = np.array(x_test)
print(x_test.shape)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], -1))

predictions = model.predict(x_test)
print('x_test.shape: ', x_test.shape)
print('predictions.shape: ', predictions.shape)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)** 2)
print('rmse: ', rmse)

train = data[:training_data_len]
print('train: ', train)
valid = data[training_data_len:]
print('valid: ', valid)
valid[['Predictions1','Predictions2','Predictions3','Predictions4','Predictions5','Predictions6']] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.plot(train['High'])
plt.plot(valid[['High','Predictions1']])
plt.legend(['Train', 'Val', 'Predictions1'], loc = 'lower right')
plt.show()

print('valid: ', valid)

new_df = getData('AAPL', 'yahoo', '2012-01-01', '2021-03-02')
print('new_df: ', new_df)
last_60_days = new_df[-60:].values
print('last_60_days: ', last_60_days)
last_60_days_scaled = scaler.transform(last_60_days)
print('last_60_days_scaled: ', last_60_days_scaled)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
print(X_test.shape)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], -1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print('pred_price: ', pred_price)

apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2021-03-04', end='2021-03-04')
print('asdsa: ', apple_quote2)