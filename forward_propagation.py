import numpy as np
import tensorflow as tf

#sample training example with 2 features
x = np.array([[2.0, 3.0]])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation = 'sigmoid', name = 'Layer1'),
    tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'Layer2')
])

output = model(x)

print(output.numpy())
