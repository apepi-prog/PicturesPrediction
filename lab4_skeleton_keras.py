from __future__ import print_function

#The two folloing lines allow to reduce tensorflow verbosity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1' # '0' for DEBUG=all [default], '1' to filter INFO msgs, '2' to filter WARNING msgs, '3' to filter all msgs

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import numpy as np



print('tensorflow:', tf.__version__)
print('keras:', tensorflow.keras.__version__)

##Uncomment the following two lines if you get CUDNN_STATUS_INTERNAL_ERROR initialization errors.
## (it happens on RTX 2060 on room 104/moneo or room 204/lautrec) 
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)



#load (first download if necessary) the MNIST dataset
# (the dataset is stored in your home direcoty in ~/.keras/datasets/mnist.npz
#  and will take  ~11MB)
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train : 60000 images of size 28x28, i.e., x_train.shape = (60000, 28, 28)
# y_train : 60000 labels (from 0 to 9)
# x_test  : 10000 images of size 28x28, i.e., x_test.shape = (10000, 28, 28)
# x_test  : 10000 labels
# all datasets are of type uint8
print('x_train.shape=', x_train.shape)
print('y_test.shape=', y_test.shape)

#To input our values in our network Conv2D layer, we need to reshape the datasets, i.e.,
# pass from (60000, 28, 28) to (60000, 28, 28, 1) where 1 is the number of channels of our images
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255

print('x_train.shape=', x_train.shape)
print('x_test.shape=', x_test.shape)

num_classes = 10

#Convert class vectors to binary class matrices ("one hot encoding")
## Doc : https://keras.io/utils/#to_categorical
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

n_epochs = 10
tensorboard_cb = tensorflow.keras.callbacks.TensorBoard(log_dir="./mylogs")

#Let start our work: creating a convolutional neural network
#ajouter Dropout
model = Sequential()
model.add(Conv2D(24,
                     kernel_size=(5,5),
                     activation='relu',
                     input_shape=(28, 28, 1),
                     strides=(1,1),
                     padding='same'))
model.add(Conv2D(64,
                     kernel_size=(3,3),
                     activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.2))
# model.add(Conv2D(64,
#                      kernel_size=(3,3),
#                      activation='relu'))
# model.add(Conv2D(32,
#                      kernel_size=(9,9),
#                      activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax')) #softmax sur la dernière couche uniquement
model.summary() #trainable parameters

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_crossentropy'])

hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=n_epochs, batch_size=128, callbacks=[tensorboard_cb]) 

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
#con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir='my_logs')
# ...
# writer.addscaler('Loss/Train', train_loss, writer)

# Dropout(proportion de neurones a eteindre)
# Normalization / Standardization
