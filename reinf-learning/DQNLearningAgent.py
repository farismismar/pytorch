#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 10:09:07 2020

@author: farismismar
"""

# Deep Q-learning Agent
import random
import numpy as np
from collections import deque
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers, initializers
import tensorflow as tf
from tensorflow.compat.v1 import set_random_seed
import os

class DQNLearningAgent:
    def __init__(self, learning_rate=0.05,
                 discount_factor=0.995,
                 exploration_rate=1.0,
                 exploration_decay_rate=0.99, batch_size=32,
                 state_size=4, action_size=4, random_state=None):

        self.learning_rate = learning_rate          # alpha
        self.discount_factor = discount_factor      # gamma
        self.exploration_rate = exploration_rate    # epsilon
        self.exploration_rate_min = 0.01
        self.exploration_decay_rate = exploration_decay_rate # d

        self.state_size = state_size
        self.action_size = action_size
        self.random_state = random_state
        
        self.model = self._build_model()

        self.memory = deque(maxlen=2000)
        self.batch_size = batch_size
        self.prefer_gpu = True
        
        # Add a few lines to caputre the seed for reproducibility.
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(random_state)
        self.rng = np.random.RandomState(random_state)
        set_random_seed(random_state)
        
        self.use_cuda = len(tf.config.list_physical_devices('GPU')) > 0 and self.prefer_gpu
        self.device = "/gpu:0" if self.use_cuda else "/cpu:0"


    def prepare_agent(self, env=None):
        return    # to match the vanilla Q-learning---not needed otherwise.
    
        
    def _build_model(self):
        # Neural Net for Deep Q learning Model from state_size |S| to action_size |A|
        # Keep adding depth until the losses start to subside (from explosive) while Q increases.
        hidden_dim = 8
        
        # Ensure reproducibility
        alpha = 1. / np.sqrt(hidden_dim) 
        initializer = initializers.RandomUniform(minval=-alpha, maxval=alpha, seed=self.random_state)
        
        model = keras.Sequential(
            [
                keras.Input(shape=self.state_size),
                layers.Dense(hidden_dim, use_bias=True, activation="relu", bias_initializer='zeros', kernel_initializer=initializer),
                layers.Dense(hidden_dim, use_bias=True, activation="relu", bias_initializer='zeros', kernel_initializer=initializer),
                layers.Dense(hidden_dim, use_bias=True, activation="relu", bias_initializer='zeros', kernel_initializer=initializer),
                layers.Dense(hidden_dim, use_bias=True, activation="relu", bias_initializer='zeros', kernel_initializer=initializer),
                layers.Dense(hidden_dim, use_bias=True, activation="relu", bias_initializer='zeros', kernel_initializer=initializer),
                layers.Dense(hidden_dim, use_bias=True, activation="relu", bias_initializer='zeros', kernel_initializer=initializer),
                layers.Dense(hidden_dim, use_bias=True, activation="relu", bias_initializer='zeros', kernel_initializer=initializer),

                layers.Dense(self.action_size, activation='linear') 
            ]
        )
        
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model
    
    
    def _construct_training_set(self, replay):
        # Select states and next states from replay memory
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        
        # Predict the expected Q of current state and new state using DQN
        with tf.device(self.device):
            Q = self.model.predict(states)
            Q_new = self.model.predict(new_states)

        replay_size = len(replay)
        X = np.empty((replay_size, self.state_size))
        y = np.empty((replay_size, self.action_size))
        
        # Construct training set
        for i in np.arange(replay_size):
            state_r, action_r, reward_r, new_state_r, done_r = replay[i]
            target = Q[i]
            target[action_r] = reward_r

            if not done_r:
                target[action_r] += self.discount_factor * np.amax(Q_new[i])
            else:
                True # A placeholder
                # If done, no need to take another step, 
                # Environment will add max reward

            X[i] = state_r
            y[i] = target

        return X, y
    
        
    def remember(self, state, action, reward, next_state, done):
        # Make sure we restrict memory size to specified limit
        if len(self.memory) > 2000:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
        

    def act(self, state, reward=None):
        # Exploration/exploitation: choose a random action or select the best one.
        if np.random.uniform(0, 1) <= self.exploration_rate:
            return self.rng.randint(0, self.action_size)
       
        state = np.reshape(state, [1, self.state_size])
        #states = states[:,0]
        with tf.device(self.device):
            act_values = self.model.predict(state)
            
        return np.argmax(act_values[0])  # returns action
    
    
    def replay(self):
        batch = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch)
        
        X, y = self._construct_training_set(minibatch)
        with tf.device(self.device):
            loss = self.model.train_on_batch(X, y)
            # history = self.model.fit(X, y, epochs=1, verbose=0)
            # loss = history.history['loss']
            
        _q = np.mean(y)
        
        return [_q, loss]
    
    
    def begin_episode(self, observation):
        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_decay_rate
        if (self.exploration_rate < self.exploration_rate_min):
            self.exploration_rate = self.exploration_rate_min
            
        # return an action at random
        action = random.randrange(self.action_size)

        return action
    
    
    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
        self.model.save_weights(name)
    