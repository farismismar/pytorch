# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:13:21 2021

@author: farismismar
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.add_dll_directory("/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";   # My NVIDIA GeForce RTX 3050 Ti GPU output from line 16

import tensorflow as tf
#print(tf.config.list_physical_devices('GPU'))

import random
import numpy as np
from tensorflow.compat.v1 import set_random_seed

import time

prefer_gpu = True
seed = 0

random.seed(seed)
np_random = np.random.seed(seed)
set_random_seed(seed)

use_cuda = len(tf.config.list_physical_devices('GPU')) > 0 and prefer_gpu
device = "/gpu:0" if use_cuda else "/cpu:0"

start_time = time.time()
for i in range(10000):
    with tf.device(device):
        x = tf.ones([1, 2])
        y = tf.ones([2, 1])
        z = x * y
end_time = time.time()
print(end_time - start_time)

start_time = time.time()
for i in range(10000):
    with tf.device('/cpu:0'):
        x = tf.ones([1, 2])
        y = tf.ones([2, 1])
        z = x * y
end_time = time.time()
print(end_time - start_time)

print(z)