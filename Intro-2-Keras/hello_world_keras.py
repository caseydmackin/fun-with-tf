import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


print(tf.VERSION)
print(tf.keras.__version__)

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Categorical Classification
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))

model.evaluate(data, labels, batch_size=32)

result = model.predict(data, batch_size=32)

print(result.shape)
print(result)

