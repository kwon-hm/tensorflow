import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(tf.test.gpu_device_name())
print(tf.test.is_gpu_available())