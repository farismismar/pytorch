#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 17:29:20 2026

@author: farismismar
"""

# This is for PPO instead of the old DQN.

import random
import numpy as np
from collections import deque

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras import backend as K


class PPOLearningAgent:
    def __init__(self,
                 seed,
                 state_size=8,
                 action_size=16,
                 gamma=0.99,
                 lam=0.95,
                 clip_ratio=0.2,
                 actor_lr=3e-4,
                 critic_lr=1e-3,
                 entropy_coeff=0.01,
                 batch_size=64,
                 epochs=10):

        # Reproducibility
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self._state_size = state_size
        self._action_size = action_size

        # PPO hyperparameters
        self.gamma = gamma
        self.lam = lam  # for GAE (set to 1 for no GAE).
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.batch_size = batch_size
        self.epochs = epochs

        # Trajectory buffer (on-policy)
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        # Build networks
        self.actor = self._build_actor(actor_lr)
        self.critic = self._build_critic(critic_lr)

        gpu_available = tf.test.is_gpu_available()
        if not gpu_available:
            print("WARNING: No GPU available.  Will continue with CPU.")


    def _build_actor(self, lr):
        state_input = Input(shape=(self._state_size,))
        x = Dense(64, activation='relu')(state_input)
        x = Dense(64, activation='relu')(x)
        logits = Dense(self._action_size, activation='linear')(x)

        model = Model(state_input, logits)
        model.optimizer = Adam(lr=lr)
        return model
        

    def _build_critic(self, lr):
        state_input = Input(shape=(self._state_size,))
        x = Dense(64, activation='relu')(state_input)
        x = Dense(64, activation='relu')(x)
        value = Dense(1, activation='linear')(x)

        model = Model(state_input, value)
        model.compile(optimizer=Adam(lr=lr), loss='mse')
        return model


    def act(self, state):
        state = np.reshape(state, [1, self._state_size])
        logits = self.actor.predict(state, verbose=0)
        probs = tf.nn.softmax(logits).numpy()[0]

        action = np.random.choice(self._action_size, p=probs)
        log_prob = np.log(probs[action] + 1e-10)
        value = self.critic.predict(state, verbose=0)[0, 0]

        return action, log_prob, value


    def remember(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)


    def _compute_gae(self, last_value=0):
        advantages = np.zeros(len(self.rewards))
        gae = 0.0

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + \
                    self.gamma * (1 - self.dones[t]) * (last_value if t == len(self.rewards) - 1 else self.values[t + 1]) \
                    - self.values[t]
            gae = delta + self.gamma * self.lam * (1 - self.dones[t]) * gae
            advantages[t] = gae

        returns = advantages + np.array(self.values)
        return advantages, returns

    
    def replay(self):
        states = np.array(self.states)
        actions = np.array(self.actions)
        old_log_probs = np.array(self.log_probs)

        advantages, returns = self._compute_gae()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            idx = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = idx[start:end]

                self._train_actor(
                    states[batch_idx],
                    actions[batch_idx],
                    old_log_probs[batch_idx],
                    advantages[batch_idx]
                )

                self.critic.train_on_batch(
                    states[batch_idx],
                    returns[batch_idx]
                )

        self._clear_memory()


    def _train_actor(self, states, actions, old_log_probs, advantages):
        with tf.GradientTape() as tape:
            logits = self.actor(states, training=True)
            probs = tf.nn.softmax(logits)
            dist = tfp.distributions.Categorical(probs)

            new_log_probs = dist.log_prob(actions)
            ratio = tf.exp(new_log_probs - old_log_probs)

            clipped_ratio = tf.clip_by_value(
                ratio,
                1 - self.clip_ratio,
                1 + self.clip_ratio
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages,
                           clipped_ratio * advantages)
            )

            entropy = tf.reduce_mean(dist.entropy())
            loss = policy_loss - self.entropy_coeff * entropy

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(grads, self.actor.trainable_variables)
        )
        
        
    def _clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()


    def save(self, prefix):
        self.actor.save_weights(prefix + "_actor.h5")
        self.critic.save_weights(prefix + "_critic.h5")


    def load(self, prefix):
        self.actor.load_weights(prefix + "_actor.h5")
        self.critic.load_weights(prefix + "_critic.h5")
