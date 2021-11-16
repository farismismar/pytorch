# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 16:55:46 2020

@author: farismismar
"""


import os
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import set_random_seed
from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time
import pdb


class TimeSeriesClassifier:
    
    ver = '0.1'
    rel_date = '2021-03-03'
    
    
    def __init__(self, prefer_gpu=True, seed=None):
        
        self.use_cuda = len(tf.config.list_physical_devices('GPU')) > 0 and prefer_gpu
        self.device = "/gpu:0" if self.use_cuda else "/cpu:0"
        
        # Fix the seed to guarantee reproducibility
        self.seed = seed
        self.np_random = np.random.RandomState(seed=seed)
        set_random_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


    def __ver__(self):
        return self.ver, self.rel_date
        
    
    def load_data(self):
        df = pd.DataFrame()
        if not os.path.isdir('./dataset'):
            print('I am creating ./dataset for you.  Put your .csv files there.')
            os.mkdir('./dataset')
            return None, None
                    
        files = glob.glob('dataset/*.csv')
        for file in files:
            print(file)
            try:
                df_ = pd.read_csv(file, sep=',')
                df = pd.concat([df, df_], axis=0)
            except:
                print(f'Failed to read {file}.  Check column names.')
                continue
        
        return df
    

    def train_test_split_time(self, df, label, time_steps, train_size=0.7):
        
        y = df[label]
        X = df.drop(label, axis=1)
        
        # Split on border of time
        m = int(X.shape[0] / time_steps * train_size)
        train_rows = int(m * time_steps)
        
        test_offset = ((X.shape[0] - train_rows) // time_steps) * time_steps
        X_train = X.iloc[:train_rows, :]
        X_test = X.iloc[train_rows:(train_rows+test_offset), :]
        
        le = preprocessing.LabelEncoder()
        le.fit(y)
        encoded_y = le.transform(y)
        dummy_Y = keras.utils.to_categorical(encoded_y)
        
        Y_train = dummy_Y[:train_rows]
        Y_test = dummy_Y[train_rows:(train_rows+test_offset)]

        return X_train, X_test, Y_train, Y_test, le
    

    def _create_lstm_nn(self, input_shape, output_shape):
        
        mX, nX = input_shape
        _, nY = output_shape # this becomes the dummy coded number of beams
        
        model = keras.Sequential()
        model.add(layers.LSTM(input_shape=(mX, nX), units=nX, 
                              recurrent_dropout=0.8,
                              #dropout=0.1, # dropout does not work
                              time_major=False,
                              return_sequences=False,
                              activation='sigmoid'))

        model.add(layers.Dense(nY, activation='softmax'))
        
        optimizer = optimizers.Adam(lr=0.005)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # Reporting the number of parameters
        print(model.summary())        

        num_params = model.count_params()
        print('Number of parameters: {}'.format(num_params))
        
        return model


    def train_nn(self, X_train, X_test, Y_train, Y_test, lookahead=1, epoch_count=32, batch_size=64, scaling=True, verbose=True):
        # Store number of learning features
        mX, nX = X_train.shape
        _, nY = Y_train.shape
        
        if scaling:
            # Scale X features
            sc = MinMaxScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
        else:
            # Convert input to numpy array (if not scaling)
            X_train = X_train.values
            X_test = X_test.values

        # Now, reshape input to be 3-D: [batch, timesteps, feature]
        # https://keras.io/api/layers/recurrent_layers/lstm/        
        # This works because time_major is True, hence [timesteps, batch, feature]
        X_train = np.reshape(X_train, (-1, lookahead + 1, X_train.shape[1] // (lookahead + 1)))
        X_test = np.reshape(X_test, (-1, lookahead + 1, X_test.shape[1] // (lookahead + 1)))
    
        Y_train = np.reshape(Y_train, (-1, nY))
        Y_test = np.reshape(Y_test, (-1, nY))
        
        print('INFORMATION: Starting optimization...')
        
        model = self._create_lstm_nn(input_shape=(X_train.shape[1], X_train.shape[2]),
                                      output_shape=(X_train.shape[1], nY))
                                      
        history = model.fit(X_train, Y_train, epochs=epoch_count, batch_size=batch_size, 
                            callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=lookahead*4)],
                            validation_split=0.5,
                            verbose=verbose, shuffle=True)
        
        Y_pred_test = model.predict(X_test, batch_size=batch_size) 
            
        return history, model, Y_pred_test
    
    
    # use this to build the time lookahead
    def engineer_features(self, df, target_variable, future_lookahead=1):
    
        df_output = pd.DataFrame()
    
        df_ = df.drop(['MT'], axis=1, inplace=True)
        
        # Noting that column order is important
        # Preamble (time t) but without the target variable
        df_preamble = df_.drop(target_variable, axis=1)
        df_zeros = pd.DataFrame(np.zeros_like(df_), columns=df_.columns).add_suffix('_d')
        df_preamble = df_preamble.add_suffix('_t')
        df_preamble = pd.concat([df_preamble, df_zeros], axis=1)
        
        # Now, forecasts
        df_forecasts = []
        for i in 1 + np.arange(future_lookahead):
            df_i = df_.shift(-i).add_suffix('_t+{}'.format(i*10))
            df_diff = df_.diff(-i).add_suffix('_d+{}'.format(i*10))
            
            # Get rid of all time shifts of the target variable
            # except for the future prediction
            if i != future_lookahead:
                df_i.drop(df_i.filter(regex=target_variable).columns, axis=1, inplace=True)                

            df_i = pd.concat([df_i, df_diff], axis=1)
            df_forecasts.append(df_i)
    
        # Add preamble to df_forecasts
        df_ = df_preamble.join(df_forecasts, how='outer')
        df_output = pd.concat([df_output, df_], axis=0, ignore_index=True)
        
        # Do not drop data in a time series.  Instead, fill last value
        df_output.fillna(method='bfill', inplace=True)
        
        # Then fill the first value!
        df_output.fillna(method='ffill', inplace=True)
        
        # Drop whatever is left.
        df_output.dropna(how='any', axis=1, inplace=True)
        
        # Drop the target column
        assert(df_output.isnull().sum().sum() == 0)
        
        engineered_target_variable = f'{target_variable}_t+{10*future_lookahead}'
        
        return df_output, engineered_target_variable
    
    
    def plot_history(self, history, title):
        # Plot the losses vs epoch here        
        fig = plt.figure(figsize=(8, 5))
        
        plt.rcParams['font.family'] = "Arial"
        plt.rcParams['font.size'] = "14"
        
        plot1, = plt.plot(history.epoch, history.history['loss'], c='blue')
        plot2, = plt.plot(history.epoch, history.history['val_loss'], linestyle='--', c='blue')
        plt.grid(which='both', linestyle='--')
        
        ax = fig.gca()    
        ax_sec = ax.twinx()
        plot3, = ax_sec.plot(history.epoch, history.history['accuracy'], lw=2, c='red')
        plot4, = ax_sec.plot(history.epoch, history.history['val_accuracy'], linestyle='--', lw=2, c='red')
        
        ax.set_xlabel(r'Epoch')
        ax.set_ylabel(r'Loss')
        ax_sec.set_ylabel(r'Accuracy')
        plt.legend([plot1, plot2, plot3, plot4], [r'Training Loss', r'Validation Loss', r'Training Accuracy', r'Validation Accuracy'],
                    bbox_to_anchor=(-0.1, -0.01, 1.20, 1), bbox_transform=fig.transFigure, 
                    loc='lower center', ncol=4, mode="expand", borderaxespad=0.)
        
        plt.title(title)
        #plt.tight_layout()
        plt.savefig(f'{title}.png', dpi=fig.dpi)
        #plt.show()
        plt.close(fig)
        
    
    def run_simulation(self, df, lookahead_time, epoch_count=256, batch_size=16):
        # lookahead is the number of frames (i.e., 1 = 10 ms).  We are not doing more than 1
        label = 'target_variable'
        df_1, engineered_label = self.engineer_features(df, 
                        label, future_lookahead=lookahead_time)
        
        X_train, X_test, Y_train, Y_test, le = self.train_test_split_time(df_1, engineered_label, time_steps=lookahead_time)
        
        history, model, Y_pred_nn = self.train_nn(X_train, X_test, Y_train, Y_test, 
                                            lookahead=lookahead_time,
                                            epoch_count=epoch_count, batch_size=batch_size)
        
        predictor.plot_history(history, title='History')
        end_time = time.time()
        print("Training took {:.3f} hours.".format((end_time - start_time) / 3600.))
        
        # Reverse the encoded categories
        y_test = le.inverse_transform(np.argmax(Y_test, axis=1))
        y_pred_nn = le.inverse_transform(np.argmax(Y_pred_nn, axis=1))
        
        print('Test accuracy: {:.4f}%.'.format(100*np.mean(y_pred_nn == y_test)))
        
        df_output_ = pd.DataFrame() # X_test.copy()
        df_output_['True'] = y_test
        df_output_['Pred'] = y_pred_nn
        
        return df_output_        

  
lookahead_time = 3 # how many featuresets are needed? (3 means 4)
predictor = TimeSeriesClassifier(seed=0)
df = predictor.load_data()

start_time = time.time()
df_output = predictor.run_simulation(df, lookahead_time)
end_time = time.time()
print("Full simulation run-time {:.3f} hours.".format((end_time - start_time) / 3600.))