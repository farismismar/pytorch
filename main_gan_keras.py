# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 18:53:59 2023

@author: farismismar
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if os.name == 'nt':
    os.add_dll_directory("/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# The GPU ID to use, usually either "0" or "1" based on previous line.
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import random
import numpy as np
from tensorflow.compat.v1 import set_random_seed

from tensorflow.keras import layers

import matplotlib.pyplot as plt

import time

plt.rcParams['font.family'] = "Arial"
plt.rcParams['font.size'] = "14"

batch_size = 32
epochs = 400
lr = 1e-3

latent_dim = 2

num_samples_to_generate = 64
seed = 7

# Not working well to ensure reproducibility.
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
random_state = np.random.RandomState(seed)
set_random_seed(seed)
# tf.random.set_seed(seed)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Create the discriminator.
def make_discriminator_model():
    global latent_dim
    model = tf.keras.Sequential(name='discriminator')
    model.add(layers.Dense(units=128, input_dim=latent_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(units=64))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(units=32))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(units=16))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(1))

    return model


# Create the generator.
def make_generator_model():
    global latent_dim
    model = tf.keras.Sequential(name='generator')
    model.add(layers.Dense(units=16))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(units=16))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(units=8))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(units=latent_dim))

    return model


def discriminator_loss(real_output, fake_output):
    global cross_entropy
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
     

def generator_loss(fake_output):
    global cross_entropy
    return cross_entropy(tf.ones_like(fake_output), fake_output)
     

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled" for much
# faster execution
@tf.function
def train_step(images):
    global batch_size, latent_dim
    global generator, discriminator, generator_optimizer, discriminator_optimizer

    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(dataset, epochs, generative_samples):
    G_losses = []
    D_losses = []
    
    for epoch in range(epochs):
        start = time.time()
        
        data_size = len(train_dataset)
        
        for n, batch in dataset.enumerate():
            image_batch, _ = batch[:, :-1], batch[:, -1]
            g_loss, d_loss = train_step(image_batch)
            if n == data_size - 1:
                print(f'G loss: {g_loss:.6f}.  D loss: {d_loss:.6f}')
                G_losses.append(g_loss)
                D_losses.append(d_loss)
                
        end = time.time()
        print ('Time for epoch {} is {:.2f} sec'.format(epoch + 1, end-start))

    return G_losses, D_losses

        
def generate(model, samples):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(samples, training=False)
    
    return predictions
  
    
generator_optimizer = tf.keras.optimizers.Adam(lr)
discriminator_optimizer = tf.keras.optimizers.Adam(lr)

generator = make_generator_model()
z = tf.random.normal([batch_size, latent_dim]) # the noise to train generator
generated_image = generator(z, training=False)

discriminator = make_discriminator_model()
# decision = discriminator(generated_image)

# Train the model: training data has a dimension of 2 
# Hence the latent dimension is 2.
train_data_length = 512

train_data = np.zeros((train_data_length, 2))
train_data[:, 0] = 2 * np.pi * random_state.uniform(low=0, high=1, size=train_data_length)
train_data[:, 1] = np.sin(train_data[:, 0])
train_labels = np.zeros(train_data_length)

train_set = np.array(train_data)
train_set = np.c_[train_set, train_labels]

assert(latent_dim == train_data.shape[1])
train_dataset = tf.data.Dataset.from_tensor_slices(train_set).batch(batch_size).shuffle(train_data_length, seed=seed)

# The noise required for the generative model 
latent_space_samples = tf.random.normal([num_samples_to_generate, latent_dim])
G_losses, D_losses = train(train_dataset, epochs, latent_space_samples)

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(range(epochs), G_losses, label='Generator loss')
plt.plot(range(epochs), D_losses, label='Discriminator loss')
plt.legend()
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Binary cross-entropy loss')
plt.tight_layout()
plt.show()
plt.close(fig)

generated_samples = generate(generator, latent_space_samples)

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".", label='Generated')
plt.plot(train_data[:, 0], train_data[:, 1], ".", label='True')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
plt.close(fig)
