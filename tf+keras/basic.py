import tensorflow as tf
from tensorflow import keras
from keras import layers

import matplotlib.pyplot as plt

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[10]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

x = tf.linspace(-1, 1, 100)
y = layers.Activation('swish')(x)

plt.plot(x, y)
plt.xlabel('Input')
plt.ylabel('Output')

plt.show()

