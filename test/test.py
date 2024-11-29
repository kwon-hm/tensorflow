import tensorflow as tf
print(tf.__version__)
tf.config.experimental.list_physical_devices(device_type=None)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())