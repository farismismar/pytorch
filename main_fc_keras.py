#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 17:36:34 2020

@author: farismismar
"""

import os
from tensorflow import keras
from tensorflow.keras import layers, initializers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.compat.v1 import set_random_seed

import tensorflow as tf
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
    global seed, input_dim, hidden_1_dim, hidden_2_dim, output_dim
    
    # Uniform initializer for weight and bias
    alpha = 1. / np.sqrt(hidden_1_dim)
    initializer_1 = initializers.RandomUniform(minval=-alpha, maxval=alpha, seed=seed)
    
    alpha= 1. / np.sqrt(hidden_2_dim)
    initializer_2 = initializers.RandomUniform(minval=-alpha, maxval=alpha, seed=seed)
    
    alpha= 1. / np.sqrt(output_dim)
    initializer_3 = initializers.RandomUniform(minval=-alpha, maxval=alpha, seed=seed)
    
    model = keras.Sequential(
        [
            keras.Input(shape=input_dim),
            layers.Dense(hidden_1_dim, use_bias=True, activation="relu", kernel_initializer=initializer_1, bias_initializer='zeros'),
            layers.Dense(hidden_2_dim, use_bias=True, activation="relu", kernel_initializer=initializer_2, bias_initializer='zeros'),
            layers.Dense(output_dim, use_bias=True, activation="softmax", kernel_initializer=initializer_3, bias_initializer='zeros')
            
        ]
    )
        
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum), 
                  metrics=['accuracy', 'sparse_categorical_crossentropy'])
    
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
    
plot1, = plt.plot(history.epoch, history.history['loss'], c='blue')
plt.grid(which='both', linestyle='--')

ax = fig.gca()    
ax_sec = ax.twinx()
plot2, = ax_sec.plot(history.epoch, history.history['accuracy'], lw=2, c='red')       
ax.set_xlabel(r'Epoch')
ax.set_ylabel(r'Loss')
ax_sec.set_ylabel(r'Accuracy [%]')
plt.legend([plot1, plot2], [r'Loss', r'Accuracy [%]'],
           bbox_to_anchor=(0.1, 0.0, 0.80, 1), bbox_transform=fig.transFigure, 
           loc='lower center', ncol=3, mode="expand", borderaxespad=0.)

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