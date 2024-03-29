# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 09:25:11 2023

@author: Faris Mismar
"""

import os

# For Windows: path to NVIDIA's cudnn libraries.
if os.name == 'nt':
    os.add_dll_directory("/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # My NVIDIA GeForce RTX 3050 Ti GPU output from line 20

import tensorflow as tf
#print(tf.config.list_physical_devices('GPU'))

import random
import numpy as np
from tensorflow.compat.v1 import set_random_seed

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import time

prefer_gpu = True
n_epochs = 20
batch_size = 32

# Dimensions of X of our binary classifier
mX = 500
nX = 10

seed = 0

random.seed(seed)
np_random = np.random.RandomState(seed)
set_random_seed(seed)

use_cuda = len(tf.config.list_physical_devices('GPU')) > 0 and prefer_gpu
device = "/gpu:0" if use_cuda else "/cpu:0"


def create_mlp(width, depth, input_dim):
    global seed
    
    output_dim = 1
    hidden_dim = 10
    learning_rate = 0.01
    momentum = 0.5
    
    model = keras.Sequential()
    model.add(layers.Dense(units=hidden_dim, input_dim=input_dim, use_bias=True, activation="relu", kernel_initializer='random_uniform', bias_initializer='zeros'))
    for depth_count in range(1, depth):
        model.add(layers.Dense(units=hidden_dim, use_bias=True, activation="relu", bias_initializer='zeros'))
        
    model.add(layers.Dense(units=output_dim, use_bias=True, activation="sigmoid", bias_initializer='zeros'))

    model.compile(loss='binary_crossentropy', 
                  optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum), 
                  metrics=['accuracy', 'binary_crossentropy'])
    
    return model


X, y = make_classification(n_samples=mX, n_features=nX, random_state=np_random)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=np_random)

print(f'Using {device}.')
 
# This shows that the runtime is const in width
t_w = []
W = [5, 10, 15, 20, 25, 30]
for w in W:
    model = KerasClassifier(build_fn=create_mlp, input_dim=nX, width=w, 
                            depth=10, epochs=n_epochs, batch_size=batch_size, verbose=0)

    for iteration in range(40):
        start_time = time.time()
        with tf.device(device):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        end_time = time.time()
    
    run_time = (end_time - start_time)
    
    t_w.append(run_time)
    

# This shows that the runtime is linear in the depth.
t_d = []
D = [5, 10, 15, 20, 25, 30]
for d in D:
    model = KerasClassifier(build_fn=create_mlp, input_dim=nX, width=5, 
                            depth=d, epochs=n_epochs, batch_size=batch_size, verbose=0)

    for iteration in range(20):
        start_time = time.time()
        with tf.device(device):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        end_time = time.time()
    
    run_time = (end_time - start_time)
    
    t_d.append(run_time)
    

# Test interaction
x = W
y = D

xx, yy = np.meshgrid(x, y)

zz = []
for w in xx[0,:]:
    for d in yy[:,0]:
        model = KerasClassifier(build_fn=create_mlp, input_dim=nX, width=w, 
                                depth=d, epochs=n_epochs, batch_size=batch_size, verbose=0)

        for iteration in range(40):
            print(f'Iter: {iteration} for width {w} and depth {d}.')
            start_time = time.time()
            with tf.device(device):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            end_time = time.time()
        
        run_time = (end_time - start_time)
        
        zz.append(run_time)

zz = np.reshape(zz, xx.shape)

#######################################################################
# Generate plots
fig = plt.contourf(x, y, zz)
plt.axis('scaled')
plt.colorbar()
plt.show()

ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(x, y, zz, cmap='autumn', antialiased=True, cstride=1, rstride=1)
ax.set_xlabel('Width')
ax.set_ylabel('Depth')
ax.set_zlabel('Run time')
plt.show()

fig = plt.figure(figsize=(8,5))
ax = fig.gca()
plot1, = ax.plot(W, t_w, label='Width')
plot2, = ax.plot(D, t_d, label='Depth')
ax.legend(handles=[plot1, plot2])
plt.xlabel('Parameter')
plt.ylabel('Run time')
plt.grid()
plt.tight_layout()
plt.show()
plt.close(fig)



