#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 09:51:50 2024

@author: farismismar
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, losses

from tensorflow.keras.models import Model


seed = 7
latent_dim = 128
batch_size = 32
max_epoch = 1000

np_random = np.random.RandomState(seed=seed)

class Denoise(Model):
    def __init__(self, latent_dim, shape, seed=None):
      # Reproducibility not working well.
      # os.environ['PYTHONHASHSEED'] = str(seed)
      random.seed(seed)
      random_state = np.random.RandomState(seed)
      tf.random.set_seed(seed) # sets global random seed
      # tf.keras.utils.set_random_seed(seed)
      
      super(Denoise, self).__init__()
      
      self.latent_dim = latent_dim
      self.shape = shape
      
      # Compresses
      self.encoder = tf.keras.Sequential(name='encoder')
      self.encoder.add(layers.Flatten())
      self.encoder.add(layers.Dense(16, activation='relu'))
      self.encoder.add(layers.Dense(64, activation='relu'))
      self.encoder.add(layers.Dense(latent_dim, activation='relu'))
      
      # self.encoder.add(layers.Conv2D(32, (2, 2), input_shape=shape,
      #                                strides=(2,2), activation='relu', padding='same'))
      # self.encoder.add(layers.MaxPooling2D((2,2), padding='same'))
      # self.encoder.add(layers.Conv2D(32, (latent_dim, latent_dim), strides=(2,2), activation='relu', padding='same'))
      # self.encoder.add(layers.MaxPooling2D((2,2), padding='same'))
      
      # Decompresses
      self.decoder = tf.keras.Sequential(name='decoder')
      self.decoder.add(layers.Dense(32, activation='relu'))
      self.decoder.add(layers.Dense(tf.math.reduce_prod(shape), 
                                    activation='sigmoid'))
      self.decoder.add(layers.Reshape(shape))
      
      # self.decoder.add(layers.Conv2DTranspose(32, (2, 2), strides=(2,2), activation='relu', padding='same'))
      # self.decoder.add(layers.Conv2DTranspose(32, (2, 2), strides=(2,2), activation='relu', padding='same'))
      # self.decoder.add(layers.Conv2D(1, (2, 2), activation='sigmoid'))
      
      
  # def __init__(self, latent_dim, shape):
  #   super(Denoise, self).__init__()
  #   self.encoder = tf.keras.Sequential([
  #     layers.Flatten(),
  #     layers.Dense(latent_dim, activation='relu'),
  #   ])
  #   self.decoder = tf.keras.Sequential([
  #     layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
  #     layers.Reshape(shape)
  #   ])

  #   # self.encoder = tf.keras.Sequential([
  #   #   layers.Input(shape=(28, 28, 1)),
  #   #   layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
  #   #   layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

  #   # self.decoder = tf.keras.Sequential([
  #   #   layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
  #   #   layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
  #   #   layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded



# Denoise cannot handle negative values, somehow!
t = np.linspace(0, 2, 200)
x_train = np.sin(2*np.pi*t) ** 2

noise = np_random.normal(loc=0, scale=0.05, size=x_train.shape)
x_train_noisy = x_train + noise


shape = x_train.shape[1:]

autoencoder = Denoise(latent_dim, shape, seed)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
  
history = autoencoder.fit(x_train, x_train_noisy,
                         epochs=max_epoch, batch_size=batch_size,
                         callbacks=[callback])

x_test_noisy = x_train_noisy
encoded_x = autoencoder.encoder(x_test_noisy).numpy()
denoised_x = autoencoder.decoder(encoded_x).numpy()

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(t, x_train, '--b', label='orig')
ax.plot(t, x_train_noisy, 'r', alpha=0.5, label='noisy')
ax.plot(t, denoised_x, 'k',  alpha=0.5, label='denoised')
ax.legend()
ax.grid(True)
plt.show()
plt.close(fig)
