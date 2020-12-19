#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 10:09:07 2020

@author: farismismar
"""

import numpy as np

# This is not a Deep Q Learning Agent
# It is meant to replace a one-liner in the main driver to compare a DQN
# with a tabular Q-learner, hence the extra dummy variables/methods.

# Following from
# https://github.com/YuriyGuts/cartpole-q-learning/blob/master/cartpole.py

class QLearningAgent:
    def __init__(self, learning_rate=0.2,
                 discount_factor=0.995,
                 exploration_rate=1.0,
                 exploration_decay_rate=0.99, batch_size=32,
                 state_size=4, action_size=3, random_state=None):
        
        self.learning_rate = learning_rate          # alpha
        self.discount_factor = discount_factor      # gamma
        self.exploration_rate = exploration_rate    # epsilon
        self.exploration_rate_min = 0.010
        self.exploration_decay_rate = exploration_decay_rate # d
        self.state = np.zeros(state_size, dtype=int)
        self.action = 0
        self.batch_size = batch_size # dummy variable -- does nothing
        self.action_size = action_size        
        self.state_size = state_size
        
        self.memory = [] # another useless variable
        
        # Add a few lines to capture the seed for reproducibility.
        self.rng = np.random.RandomState(random_state)


    def prepare_agent(self, env):
        # check the site distance configuration in the environment
        self._state_bins = [
            # User X - serv
            self._discretize_range(-env.cell_radius, env.cell_radius, self.state_size),
            # User Y - serv
            self._discretize_range(-env.cell_radius, env.cell_radius, self.state_size),
            # Serving BS power.
            self._discretize_range(0, env.max_tx_power, self.state_size),
            # Beamforming
            self._discretize_range(0, env.M_ULA, self.state_size),
        ]
        
        # Create a clean Q-Table.
        self._max_bins = max(len(bin) for bin in self._state_bins)
        num_states = (self._max_bins + 1) ** len(self._state_bins)
        self.q = np.zeros(shape=(num_states, self.action_size))
    
    
        
    def begin_episode(self, observation):
        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_decay_rate
        if (self.exploration_rate < self.exploration_rate_min):
            self.exploration_rate = self.exploration_rate_min
    
        self.state = self._build_state(observation)
        
        return np.argmax(self.q[self.state, :]) # returns the action with largest Q
        
    
    def act(self, observation, reward):
        next_state = self._build_state(observation) # need to be integer
        
        state = self.state
        if isinstance(state, np.ndarray):
            state = state[0]
        # Exploration/exploitation: choose a random action or select the best one.
        enable_exploration = (1 - self.exploration_rate) <= self.rng.uniform(0, 1)
        if enable_exploration:
            next_action = self.rng.randint(0, self.action_size)
        else:
            next_action = np.argmax(self.q[next_state])
        
        # Learn: update Q-Table based on current reward and future action.
        self.q[state, self.action] += self.learning_rate * \
            (reward + self.discount_factor * max(self.q[next_state, :]) - self.q[state, self.action])
    
        self.state = next_state
        self.action = next_action
        return next_action


    def replay(self):
        loss = 0.0
        return [self.q.mean(), loss]
      
    def remember(self, prev_observation, action, reward, observation, done):
        return # this is a dummy function for compatibility

    # Private members:
    def _build_state(self, observation):
        # Discretize the observation features and reduce them to a single integer.
        state = sum(
            self._discretize_value(feature, self._state_bins[i]) * ((self._max_bins + 1) ** i)
            for i, feature in enumerate(observation)
        )
        return state
    
    def _discretize_value(self, value, bins):
        return np.digitize(x=value, bins=bins)
    
    def _discretize_range(self, lower_bound, upper_bound, num_bins):
        return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]
    
 