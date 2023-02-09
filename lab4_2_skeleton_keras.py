from __future__ import print_function

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from numpy.random import seed
import matplotlib.pyplot as plt
import numpy as np


print('tensorflow:', tf.__version__)
print('keras:', tensorflow.keras.__version__)

##Uncomment the following two lines if you get CUDNN_STATUS_INTERNAL_ERROR initialization errors.
## (it happens on RTX 2060 on room 104/moneo or room 204/lautrec) 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#load (first download if necessary) the CIFAR10 dataset
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10

#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255

print('x_train.shape=', x_train.shape)
print('x_test.shape=', x_test.shape)

#Convert class vectors to binary class matrices ("one hot encoding")
## Doc : https://keras.io/utils/#to_categorical
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

# Display one image and corresponding label 
# plt.imshow(x_train[2].reshape(32,32,3))
# plt.axis("off")
# plt.show()

x_train = np.concatenate((x_train, x_train), axis=0)
y_train = np.concatenate((y_train, y_train), axis=0)

print(x_train.shape)
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
#   tf.keras.layers.RandomZoom(-.2, -.2),
#   tf.keras.layers.RandomContrast(0.4),
])

# DATA Augmentation example
plt.figure(figsize=(10, 10))
for i in range(9):
    augmented_image = data_augmentation(x_train[7])
    print(augmented_image.shape)
    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(augmented_image)
plt.show()

print("After augmentation")
print('x_train.shape=', x_train.shape)

# Transfer training with resnet :
input_tens = tensorflow.keras.Input(shape=(32, 32, 3))
res_model = tensorflow.keras.applications.ResNet50(include_top=False,
                                                    weights="imagenet",
                                                    input_tensor=input_tens)
res_model.summary()

#Let start our work: creating a convolutional neural network
tensorboard_cb = tensorflow.keras.callbacks.TensorBoard(log_dir="./mylogs")
n_epochs = 40
model = Sequential()
model.add(data_augmentation)
model.add(res_model)
model.add(Flatten())
# model.add(Conv2D(256,
#                      kernel_size=(3,3),
#                      activation='relu',
#                      input_shape=(32, 32, 3),
#                      strides=(1,1),
#                      padding='same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())

# model.add(Conv2D(128,
#                      kernel_size=(3,3),
#                      activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Conv2D(64,
#                      kernel_size=(3,3),
#                      activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
# model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', 
            metrics=['accuracy', 'categorical_crossentropy'])

hist = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                epochs=n_epochs, batch_size=128, callbacks=[tensorboard_cb]) 

scores = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print("Neural network final accuracy: %.2f%%" % (scores[1]*100))
print('Neural network final loss: ', (scores[0]))

X = list(range(0, n_epochs))
fig, axs = plt.subplots(2,2)
axs[0][0].plot(X, acc, color='blue', linewidth= 1.5, linestyle= '-')
axs[0][1].plot(X, val_acc, color='skyblue', linewidth= 1.5, linestyle= '-')
axs[1][0].plot(X, loss, color='red', linewidth= 1.5, linestyle= '-')
axs[1][1].plot(X, val_loss, color='salmon', linewidth= 1.5, linestyle= '-')

axs[1][0].set(xlabel='Train', ylabel='Loss')
axs[0][0].set(ylabel='Accuracy')
axs[1][1].set(xlabel='Validation')
plt.show()

#matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)) #doesn't work
#print(matrix)

#TODO find 10 worst images

