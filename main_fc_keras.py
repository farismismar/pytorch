#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:36:34 2020

@author: farismismar
"""

import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.compat.v1 import set_random_seed

import numpy as np
import matplotlib.pyplot as plt

import time

n_epochs = 100
learning_rate = 0.01
momentum = 0.5
batch_size = 4
prefer_gpu = True

input_dim = 28 ** 2 # MNIST input size
hidden_1_dim = 32
hidden_2_dim = 32
output_dim = 10

os.chdir('/Users/farismismar/Desktop')

use_cuda = len(tf.config.list_physical_devices('GPU')) > 0 and prefer_gpu
device = "/gpu:0" if use_cuda else "/cpu:0"

# Fix the seed to guarantee reproducibility
seed = 0
set_random_seed(seed)
np.random.seed(seed)

# Load MNIST as Numpy arrays
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# Note: If this fails due to certificate verify failure, sudo -H pip3 install --upgrade certifi

# Must normalize the data for Keras in [0,1]---not required for PyTorch
X_train = X_train.astype("float32") / 255.
X_test = X_test.astype("float32") / 255.

X_train = X_train.reshape(X_train.shape[0], -1) # reshape
X_test = X_test.reshape(X_test.shape[0], -1) # reshape


def create_mlp():
    global seed, hidden_1_dim, hidden_2_dim
  
    model = Sequential()
    model.add(Dense(units=hidden_1_dim, input_dim=input_dim, use_bias=True))
    model.add(Activation('relu'))
    model.add(Dense(units=hidden_2_dim, input_dim=hidden_1_dim, use_bias=True))
    model.add(Activation('relu'))
    model.add(Dense(units=output_dim, input_dim=hidden_2_dim))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=SGD(learning_rate=learning_rate, momentum=momentum), 
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])
    
    alpha = 1. / np.sqrt(hidden_1_dim)
    model.layers[0].weights[0] = np.random.uniform(low=-alpha, high=alpha, size=hidden_1_dim)
    model.layers[0].weights[1] = np.random.uniform(low=-alpha, high=alpha, size=hidden_1_dim) # bias
    alpha = 1. / np.sqrt(hidden_2_dim)
    model.layers[2].weights[0] =  np.random.uniform(low=-alpha, high=alpha, size=hidden_2_dim)
    model.layers[2].weights[1] = np.random.uniform(low=-alpha, high=alpha, size=hidden_2_dim)  
    alpha = 1. / np.sqrt(output_dim)
    model.layers[4].weights[0] = np.random.uniform(low=-alpha, high=alpha, size=output_dim)
    model.layers[4].weights[1] = np.random.uniform(low=-alpha, high=alpha, size=output_dim)
    
    return model

# Early stopping condition
es = EarlyStopping(monitor='accuracy', mode='auto', verbose=1, min_delta=0.0001, patience=4)

model = KerasClassifier(build_fn=create_mlp, verbose=1, callbacks=es,
                         epochs=n_epochs, batch_size=batch_size)
start_time = time.time()
with tf.device(device):
    history = model.fit(X_train, y_train)
end_time = time.time()

print('Training time: {:.2f} mins.'.format((end_time - start_time) / 60.))

# Plot the losses vs epoch here
fig = plt.figure(figsize=(8, 5))
plt.clf()
plt.plot(history.epoch, history.history['loss'], marker='o', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Performance')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close(fig)

fig = plt.figure(figsize=(8, 5))
plt.clf()
plt.plot(history.epoch, history.history['accuracy'], color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')
plt.title('Training Performance')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close(fig)

# Testing
with tf.device(device):
    y_pred = model.predict(X_test)
    loss, acc, _ = model.model.evaluate(X_test, y_test)
    
print('Test: Loss {:.4f}, Acc: {:.4f}'.format(loss, acc))

# Reporting the number of parameters
num_params = model.model.count_params()
print('Number of parameters: {}'.format(num_params))