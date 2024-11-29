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

df = getData('AAPL', 'yahoo','2012-01-01', '2021-02-28')
print(df)
print(df.shape)

# plt.figure(figsize=(16,8))
# plt.title('Close Price History')
# plt.plot(df['Close'])
# plt.xlabel('Date', fontsize=12)
# plt.ylabel('Close Price USD', fontsize=12)
# plt.show()

data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .95)
print('training_data_len: ', training_data_len)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print('scaled_data: ', scaled_data)

train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  if i<=60:
    print('x_train: ', x_train)
    print('y_train: ', y_train)
    print()

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

model = keras.models.Sequential()
model.add(keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(keras.layers.LSTM(50, return_sequences=False))
model.add(keras.layers.Dense(25))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(
  x_train, 
  y_train, 
  batch_size=1, 
  epochs=100
)

test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)** 2)
print('rmse: ', rmse)

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
plt.show()

print('valid: ', valid)

apple_quote = getData('AAPL', 'yahoo', '2012-01-01', '2021-12-31')
print('apple_quote: ', apple_quote)
new_df = apple_quote.filter(['Close'])
print('new_df: ', new_df)
last_60_days = new_df[-60:].values
print('last_60_days: ', last_60_days)
last_60_days_scaled = scaler.transform(last_60_days)
print('last_60_days_scaled: ', last_60_days_scaled)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print('pred_price: ', pred_price)

apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2021-03-04', end='2021-03-04')
print('asdsa: ', apple_quote2['Close'])