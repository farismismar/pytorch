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

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler

import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time
import pdb


class TimeSeriesClassifier:    
    ver = '0.5'
    rel_date = '2022-02-06'
    
    
    def __init__(self, prefer_gpu=True, seed=None):
        
        self.use_cuda = len(tf.config.list_physical_devices('GPU')) > 0 and prefer_gpu
        self.device = "/gpu:0" if self.use_cuda else "/cpu:0"
        
        # Fix the seed to guarantee reproducibility
        self.seed = seed
        self.reset_seed()


    def __ver__(self):
        return self.ver, self.rel_date
        
    def reset_seed(self):
        seed = self.seed
        self.np_random = np.random.RandomState(seed=seed)
        set_random_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        return self
    
    
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
    

    def train_test_split_time(self, df, label, time_steps, train_size, rebalance=True):
        # Avoid truncating training data or test data...
        # time_steps is 12 if you want to predict from data representing months in a year
        # time_steps is 7 if you want to predict from data representing days of a week
        # time_steps is 1 if you want to predict from data representing samples
        y = df[label] # must be categorical
        X = df.drop(label, axis=1)
        
        # Balance through ROS (SMOTE did not work due to n_neigh > n_samples)
        if rebalance:
            try:
                print('Oversampling to balance classes...')
                ros = RandomOverSampler(random_state=self.seed)
                X, y = ros.fit_resample(X, y)
            except Exception as e:
                print(f'WARNING: Oversampling failed due to {e}.  No rebalancing performed.')
                pass
            
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

        return X_train, X_test, Y_train, Y_test, le, X_test.index
    

   def _create_lstm_nn(self, input_shape, output_shape, depth=8, width=16):
        mX, nX = input_shape
        _, nY = output_shape # this becomes the dummy coded number of beams
        
        model = keras.Sequential()
        model.add(layers.LSTM(input_shape=(mX, nX), recurrent_dropout=0.2,
                              units=width, return_sequences=True, 
                              activation="tanh",
                              recurrent_activation="sigmoid"))

        for hidden in np.arange(depth):
            model.add(layers.Dense(width, activation='relu'))
        
        model.add(layers.Dropout(0.1))
        model.add(layers.LSTM(units=width, recurrent_dropout=0.5, return_sequences=False, 
                              activation="tanh", recurrent_activation="sigmoid"))
        model.add(layers.Dense(nY, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Reporting the number of parameters
        print(model.summary())
    
        num_params = model.count_params()
        print('Number of parameters: {}'.format(num_params))
        
        return model

    
    def train_nn(self, X_train, X_test, Y_train, Y_test, lookbacks, epoch_count, batch_size, callback=None, verbose=True):
        # Store number of learning features
        mX, nX = X_train.shape
        _, nY = Y_train.shape
        
        # Scale X features
        sc = MinMaxScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Now, reshape input to be 3-D: [timesteps, batch, feature]
        X_train = np.reshape(X_train, (-1, lookbacks + 1, X_train.shape[1] // (lookbacks + 1)))
        X_test = np.reshape(X_test, (-1, lookbacks + 1, X_test.shape[1] // (lookbacks + 1)))
    
        Y_train = np.reshape(Y_train, (-1, nY))
        Y_test = np.reshape(Y_test, (-1, nY))
        
        print('INFORMATION: Starting optimization...')
        
        model = self._create_lstm_nn(input_shape=(X_train.shape[1], X_train.shape[2]),
                                      output_shape=(X_train.shape[1], nY))
        
        #patience = 2 * lookbacks
        callback_list = [] # EarlyStopping(monitor='val_loss', min_delta=0, patience=patience)]
        
        if callback is not None:
            callback_list = callback_list + [callback]
            
        history = model.fit(X_train, Y_train, epochs=epoch_count, batch_size=batch_size, 
                            callbacks=callback_list,
                            validation_data=(X_test, Y_test), 
                            #validation_split=0.5,
                            verbose=verbose, shuffle=True)
        
        Y_pred_test = model.predict(X_test, batch_size=batch_size) 
            
        return history, model, Y_pred_test
      
        
    def engineer_features(self, df, target_variable, lookahead, lookbacks, dropna=False):
        df_ = df.set_index('Time')
        df_y = df_[target_variable].to_frame()
        
        # This is needed in case the idea is to predict y(t), otherwise
        # the data will contain y_t in the data and y_t+0, which are the same
        # and predictions will be trivial.
        if lookahead == 0:
            df_ = df_.drop(target_variable, axis=1)
        
        df_postamble = df_.add_suffix('_t')
        df_postamble = pd.concat([df_postamble, pd.DataFrame(np.zeros_like(df_), index=df_.index, columns=df_.columns).add_suffix('_d')], axis=1)
        
        df_shifted = pd.DataFrame()
        # Noting that column order is important
        for i in range(lookbacks, 0, -1):
            df_shifted_i = df_.shift(i).add_suffix('_t-{}'.format(i*10))
            df_diff_i = df_.diff(i).add_suffix('_d-{}'.format(i*10)) # difference with previous time
            
            df_shifted = pd.concat([df_shifted, df_shifted_i, df_diff_i], axis=1)
           # df_shifted = pd.concat([df_shifted, df_shifted_i], axis=1)
    
        df_y_shifted = df_y.shift(-lookahead).add_suffix('_t+{}'.format(lookahead*10))
        df_output = pd.concat([df_shifted, df_postamble, df_y_shifted], axis=1)
        
        if dropna:
            df_output.dropna(inplace=True)
        else:
            # Do not drop data in a time series.  Instead, fill last value
            df_output.fillna(method='bfill', inplace=True)
           
            # Then fill the first value!
            df_output.fillna(method='ffill', inplace=True)
           
            # Drop whatever is left.
            df_output.dropna(how='any', axis=1, inplace=True)
       
        # Whatever it is, no more nulls shall pass!
        assert(df_output.isnull().sum().sum() == 0)
    
        engineered_target_variable = f'{target_variable}_t+{10*lookahead}'
        
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
                    bbox_to_anchor=(-0.1, -0.02, 1.20, 1), bbox_transform=fig.transFigure, 
                    loc='lower center', ncol=4, mode="expand", borderaxespad=0.)
        
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f'{title}.png', dpi=fig.dpi)
        #plt.show()
        plt.close(fig)
       
    
    def run_prediction(self, df, lookahead_time, lookbacks_time, train_size=0.7, epoch_count=256, batch_size=16):
        # lookahead is the number of frames (i.e., 1 = 10 ms).  We are not doing more than 1
        predictor.reset_seed()
        
        label = 'target_variable'
        df_1, engineered_label = self.engineer_features(df, label, lookahead=lookahead_time, lookbacks=lookbacks_time)
        
        X_train, X_test, Y_train, Y_test, le, _ = self.train_test_split_time(df_1, 
                                            engineered_label, time_steps=1, train_size=train_size, rebalance=True)
        
        history, model, Y_pred_nn = self.train_nn(X_train, X_test, Y_train, Y_test, 
                                                            lookbacks=lookbacks_time, callback=None,
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

  
lookahead_time = 0 # which feature set is being predicted? y(t - k)
lookbacks_time = 3 # use 3 previous feature sets (t-3, t-2, t-1, and t) to predict the feature set at t + 3

train_size = 0.2 
predictor = TimeSeriesClassifier(seed=0)
df = predictor.load_data()

start_time = time.time()
df_output = predictor.run_prediction(df, lookahead_time, lookbacks_time, train_size)
end_time = time.time()
print("Total run time {:.3f} hours.".format((end_time - start_time) / 3600.))
