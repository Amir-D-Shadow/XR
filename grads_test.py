import numpy as np
import tensorflow as tf

s = tf.Variable(3.0,trainable=True)
t = tf.Variable(5.0,trainable=True)

with tf.GradientTape() as tape:

    x = s + 8

    y = s*t + 1

    z = x*y + x + y + 6

    loss = 0.3*y + z

#dloss_ds:78.5 , dloss_dt: 36.9
grads =  tape.gradient(loss,[s,t])


model = tf.keras.Sequential([
    tf.keras.layers.Input((16,32)),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(16)
])


model.build()
a = model.trainable_variables
