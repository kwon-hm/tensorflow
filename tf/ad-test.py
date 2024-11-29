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
conn = mongo.conn('deviceStats')


# Download the dataset
dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
raw_data = dataframe.values
dataframe.head()

