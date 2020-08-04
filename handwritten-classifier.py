# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 04:08:56 2020

@author: Dio VB
"""


import tensorflow as tf
tf.get_logger().setLevel('ERROR')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

import matplotlib.pyplot as plt
print("Aktual:",y_train[3])
plt.axis("off")
plt.imshow(x_train[3], cmap='Greys')
print(x_train.ndim)

number_one = x_train[3].flatten()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255



from tensorflow.keras.models import Sequential,save_model,load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) 
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)

model.evaluate(x_test, y_test)


# Save the model
filepath = 'saved_model'
save_model(model, filepath)

model.summary()

# Load the model
loaded_model = load_model(
    filepath,
    custom_objects=None,
    compile=True
)
plt.axis("off")
plt.imshow(x_test[2].reshape(28, 28),cmap='Greys')
pred = loaded_model.predict(x_test[2].reshape(1, 28, 28, 1))
print("Actual:" ,pred.argmax())
