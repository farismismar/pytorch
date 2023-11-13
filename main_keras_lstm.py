# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 19:44:40 2023

@author: farismismar
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

from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import set_random_seed

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import pdb

from sklearn.preprocessing import MinMaxScaler


def _create_lstm_nn_regr(input_shape, depth=5, width=10):
    mX, nX = input_shape
    nY = 1 # do not change it
    
    model = keras.Sequential()
    model.add(layers.LSTM(units=32,
                         input_shape=(mX, nX),
                         return_sequences=True))
    
    for hidden in range(depth):
        model.add(layers.Dense(width, activation='sigmoid'))
   
    model.add(layers.Dense(nY))
    model.compile(loss=my_loss_fn_regr, optimizer='adam', 
                  metrics=[keras.metrics.RootMeanSquaredError()])
    
    # Reporting the number of parameters
    print(model.summary())
    
    num_params = model.count_params()
    print('Number of parameters: {}'.format(num_params))
    
    return model
    

def _create_lstm_nn_classifier(input_shape, depth=5, width=10):
    mX, nX = input_shape
    nY = 1 # do not change it
    
    model = keras.Sequential()
    model.add(layers.LSTM(units=32,
                         input_shape=(mX, nX),
                         return_sequences=True))
    
    for hidden in range(depth):
        model.add(layers.Dense(width, activation='sigmoid'))
   
    model.add(layers.Dense(nY, activation='softmax'))
    
    model.compile(loss=my_loss_fn_classifier, optimizer='adam', 
                  metrics=[tf.keras.metrics.Precision()]) # Never use accuracy!
    
    # Reporting the number of parameters
    print(model.summary())
    
    num_params = model.count_params()
    print('Number of parameters: {}'.format(num_params))
    
    return model
    

# This is the objective function that the optimizer aims at improving.
def my_loss_fn_regr(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

    
def my_loss_fn_classifier(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return bce(y_true, y_pred)


def deep_predict(df, label, train_split, is_classifier=False, epoch_count=10, batch_size=8, prefer_gpu=True, verbose=True, seed=None):
    cell_count = df.reset_index()['batch_id'].nunique()
        
    # Fill in columnar mean
    if (df.isnull().sum().sum() > 0):
        print("Warning:  Missing values detected and are imputed by mean.")
        df.fillna(df.mean(), inplace=True)
    
    y = df[label]
    X = df.drop(label, axis=1)
    
    m = (int(X.shape[0] * train_split) // cell_count) * cell_count
    X_train = X.iloc[:m, :]
    X_test = X.iloc[m:, :]
    y_train = y.iloc[:m]
    y_test = y.iloc[m:]

    idx = y_test.index
    
    # Scale only if a classifier
    if is_classifier:
        sc = MinMaxScaler()
    
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    else:
        X_train = X_train.values
        X_test = X_test.values
        
    use_cuda = len(tf.config.list_physical_devices('GPU')) > 0 and prefer_gpu
    device = "/gpu:0" if use_cuda else "/cpu:0"

    if seed is not None:
        random.seed(seed)
        random_state = np.random.RandomState(seed)
        set_random_seed(seed)
        keras.utils.set_random_seed(seed)
        # tf.config.experimental.enable_op_determinism()
    else:
        random_state=None

    # Now, reshape input to be 3-D: [timesteps, batch size per step, feature ct]
    X_train = X_train.reshape(-1, cell_count, X_train.shape[1])
    X_test = X_test.reshape(-1, cell_count, X_test.shape[1])
    
    # The output is 2-D since each label corresponds to one batch of X.
    Y_train = y_train.values.reshape(-1, cell_count)
    Y_test = y_test.values.reshape(-1, cell_count)

    if is_classifier:
        model = _create_lstm_nn_classifier(input_shape=(X_train.shape[1], X_train.shape[2]))
    else:
        model = _create_lstm_nn_regr(input_shape=(X_train.shape[1], X_train.shape[2]))

    with tf.device(device):
        history = model.fit(X_train, Y_train, epochs=epoch_count, batch_size=batch_size, 
                              validation_data=(X_test, Y_test), 
                              verbose=verbose)
        
    # Testing
    with tf.device(device):
        Y_pred = model.predict(X_test, batch_size=batch_size)
        loss, score = model.evaluate(X_test, Y_test)

    # Reconstruct Y_pred as y_
    y_pred = Y_pred.ravel()
    y_pred = pd.Series(y_pred, index=idx, name=f'{label}_pred')
                          
    return history, y_test, y_pred, score, model
