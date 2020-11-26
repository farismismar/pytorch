#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 21:43:37 2020

@author: farismismar
"""

import os
import torch
import torch.nn as nn
import torchvision

import numpy as np
import matplotlib.pyplot as plt

import time

n_epochs = 15
learning_rate = 0.01
momentum = 0.5
batch_size = 64
prefer_gpu = True

input_dim = (28, 28, 1) # MNIST input size
filter_1_dim = 32
filter_2_dim = 64
output_dim = 10

accuracy_threshold = 0.99

os.chdir('/Users/farismismar/Desktop')

use_cuda = torch.cuda.is_available() and prefer_gpu
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.enabled = use_cuda

# Fix the seed to guarantee reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

# Load MNIST using Torchvision
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=True, download=True, 
                               transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(), 
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])), batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=False, download=True, 
                               transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(), 
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])), batch_size=batch_size, shuffle=True)

# This is torch.Size([64, 1, 28, 28])
# What is 32 * 7 * 7 exactly?
# Formula given two tensors and a Conv2d layer
# https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
# https://pytorch.org/docs/master/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
# An easier way is to create a submodel up to nn.Flatten() and check the shape:
# submodel(data).shape

# Convert the tuple into a single number
input_dim_1d = np.prod(input_dim)

model = torch.nn.Sequential(
            # Features
            nn.Conv2d(in_channels=1, out_channels=filter_1_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=filter_1_dim, out_channels=filter_2_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            # Classification
            nn.Dropout(p=0.2),
            nn.Linear(filter_2_dim * 7 * 7, input_dim_1d), # 64, 1, 28, 28 becomes 64, 32, 7, 7
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(input_dim_1d, input_dim_1d, bias=True),
            nn.ReLU(),
            nn.Linear(input_dim_1d, output_dim, bias=True),
            nn.Softmax(dim=1)
        ).to(device)

# PyTorch sets weights from a Uniform distribution automatically.
# model[0].weight

# GD with adaptive moments
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()

# Training
history = {'epoch': [], 'loss': [], 'accuracy': []}

start_time = time.time()
for epoch in np.arange(n_epochs):
    model.train()
    epoch_loss = 0
    correct_train = total_train = 0 
    # Iterate on batches to introduce Stochastic GD
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # TODO: Image augmentation for data
    
        optimizer.zero_grad() # minimize
        output = model.forward(data)
        loss = criterion(output, target)
        loss.backward() # backward propagation
        optimizer.step() # update optimizer
        
        epoch_loss += batch_size * loss.item() # since loss reduction is by mean
        
        # accuracy
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max probability
        correct_train += pred.eq(target.view_as(pred)).sum().item()
        total_train += target.nelement()
        train_accuracy = correct_train / total_train
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f},\tAcc: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), train_accuracy))
        
    # Update the information for the training losses
    accuracy_ = correct_train / total_train
    history['epoch'].append(epoch)
    history['loss'].append(epoch_loss / len(train_loader.dataset))
    history['accuracy'].append(accuracy_)
    
    if (accuracy_ >= accuracy_threshold):
        print('Target accuracy reached at current epoch.  Stopping.')
        break

end_time = time.time()

print('Training time: {:.2f} mins.'.format((end_time - start_time) / 60.))

# Plot the losses vs epoch here
fig = plt.figure(figsize=(8, 5))
plt.clf()
plt.plot(history['epoch'], history['loss'], marker='o', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Performance')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close(fig)

fig = plt.figure(figsize=(8, 5))
plt.clf()
plt.plot(history['epoch'], history['accuracy'], color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')
plt.title('Training Performance')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close(fig)

# Testing
model.eval()
test_loss = 0
correct_test = 0 
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model.forward(data)
        loss = criterion(output, target)
        test_loss += batch_size * loss.item()
        
        # accuracy
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max probability
        correct_test += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest: Loss: {:.4f}, Acc: {:.4f}'.format(
        test_loss, correct_test / len(test_loader.dataset)))

# Reporting the number of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: {}'.format(num_params))
