#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 20:33:38 2020

@author: farismismar
"""

# Deep Q-learning Agent
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torchvision
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
        
        self.prefer_gpu = True
        
        # Add a few lines to caputre the seed for reproducibility.
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(random_state)
        self.rng = np.random.RandomState(random_state)
        torch.manual_seed(random_state)
        
        self.use_cuda = torch.cuda.is_available() and self.prefer_gpu
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        torch.backends.cudnn.enabled = self.use_cuda

        self.model, self.optimizer, self.criterion = self._build_model()

        self.memory = deque(maxlen=2000)
        self.batch_size = batch_size

    def prepare_agent(self, env=None):
        return    # to match the vanilla Q-learning---not needed otherwise.
    
        
    def _build_model(self):
        # Neural Net for Deep Q learning Model from state_size |S| to action_size |A|
        # Keep adding depth until the losses start to subside (from explosive) while Q increases.
        hidden_dim = 8
        
        model = torch.nn.Sequential(
             nn.Linear(self.state_size, hidden_dim, bias=True),
             nn.ReLU(),
             nn.Linear(hidden_dim, hidden_dim, bias=True),
             nn.ReLU(),
             nn.Linear(hidden_dim, hidden_dim, bias=True),
             nn.ReLU(),
             nn.Linear(hidden_dim, hidden_dim, bias=True),
             nn.ReLU(),
             nn.Linear(hidden_dim, hidden_dim, bias=True),
             nn.ReLU(),
             nn.Linear(hidden_dim, hidden_dim, bias=True),
             nn.ReLU(),
             nn.Linear(hidden_dim, self.action_size, bias=True),
        ).to(self.device)

        # Initialize weights and biases        
        model.apply(self._initialize_model)
        
        # GD with adaptive moments
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss(reduction='mean')

        return model, optimizer, criterion
    
    
    def _initialize_model(self, m):
        classname = m.__class__.__name__
        
        # If the model is linear, initailize its weights
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            alpha = 1. / np.sqrt(n)
            m.weight.data.uniform_(-alpha, alpha)
            m.bias.data.fill_(0.)
            
            
    def _construct_training_set(self, replay):
        # Select states and next states from replay memory
        # state, action, reward, new_state, done
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        
        # Predict the expected Q of current state and new state using DQN
        Q = self._predict(states)
        Q_new = self._predict(new_states)
        
        replay_size = len(replay)
        X = np.empty((replay_size, self.state_size))
        y = np.empty((replay_size, self.action_size))
        
        # Construct training set
        for i in np.arange(replay_size):
            state_r, action_r, reward_r, new_state_r, done_r = replay[i]
            target = Q[i]
            target[action_r] = reward_r

            if not done_r:
                target[action_r] += self.discount_factor * torch.max(Q_new[i]) # np.amax
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
        act_values = self._predict(state)
            
        return np.argmax(act_values[0])  # returns action
    
    
    def replay(self):
        batch = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch)
        
        X, y = self._construct_training_set(minibatch)
        loss = self._train(X, y, epochs=1)
        
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
    
    
    def clear(self):
        self.memory.clear()
        
        
    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
        self.model.save_weights(name)

    
    def _train(self, X, y, epochs=1, verbose=0):
        # trains on one epoch and generates loss
        X = torch.Tensor(X).type(torch.FloatTensor)
        y = torch.Tensor(y).type(torch.FloatTensor)

        # Training
        history = {'epoch': [], 'loss': [], 'score': []}
  
        for epoch in np.arange(epochs):
            self.model.train()
            
            # Working on a minibatch (to introduce Stochastic GD)            
            data, target = X.to(self.device), y.to(self.device)
                    
            self.optimizer.zero_grad() # minimize
            output = self.model.forward(data)
            loss = self.criterion(output, target)
            loss.backward() # backward propagation
            self.optimizer.step() # update optimizer
            
            epoch_loss = loss.item()
            
            if verbose > 0:
                print('Train Epoch: \tLoss: {:.2f}'.format(loss.item()))
                    
            # Update the information for the training losses
            history['epoch'].append(epoch)
            history['loss'].append(epoch_loss)
                
        return history['loss']
    
    
    def _predict(self, X):
        # Testing
        self.model.eval()
        with torch.no_grad():
           data = torch.from_numpy(X).type(torch.FloatTensor).to(self.device)
           y_pred = self.model.forward(data) # predicts

        return y_pred