#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 08:56:31 2020

@author: farismismar
"""

import os
import torch
import torch.nn as nn
import torchvision

import numpy as np
import matplotlib.pyplot as plt

import time

n_epochs = 100
learning_rate = 0.01
momentum = 0.5
batch_size = 4
prefer_gpu = True

loss_threshold = 200

input_dim = 28 ** 2
hidden_1_dim = 32
hidden_2_dim = 32
output_dim = 10

os.chdir('/Users/farismismar/Desktop')

use_cuda = torch.cuda.is_available() and prefer_gpu
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.enabled = use_cuda

# Fix the seed to guarantee reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)


# Initialize the weights for better grad descent behavior
def _initialize_model(m):
    classname = m.__class__.__name__
    
    # If the model is linear, initailize its weights
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        alpha = 1. / np.sqrt(n)
        m.weight.data.uniform_(-alpha, alpha)
        m.bias.data.fill_(0.)


# Load MNIST using Torchvision (clearly the target variable must be continuous)
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

# Construct the DNN 
model = torch.nn.Sequential(
         nn.Linear(input_dim, hidden_1_dim, bias=True),
         nn.ReLU(),
         nn.Linear(hidden_1_dim, hidden_2_dim, bias=True),
         nn.ReLU(),
         nn.Linear(hidden_1_dim, output_dim, bias=True),
        ).to(device)

# Initialize weights and biases        
model.apply(_initialize_model)
          
# GD with adaptive moments
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()

# Training
history = {'epoch': [], 'loss': []}

start_time = time.time()
for epoch in np.arange(n_epochs):
    model.train()
    epoch_loss = 0
    correct_train = total_train = 0 
    # Iterate on batches to introduce Stochastic GD
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.shape[0], -1) # reshape

        optimizer.zero_grad() # minimize
        output = model.forward(data) # a prediction
        loss = criterion(output, target)
        loss.backward() # backward propagation
        optimizer.step() # update optimizer
        
        epoch_loss += batch_size * loss.item() # since loss reduction is by mean
        
        if batch_idx % 50 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ', end='')
            print('({:.0f}%)]\tLoss: {:.2f}'.format(100. * batch_idx / len(train_loader), loss.item()))
        
    # Update the information for the training losses
    history['epoch'].append(epoch)
    history['loss'].append(epoch_loss / len(train_loader.dataset))
    
    if (epoch_loss <= loss_threshold):
        print('Target accuracy reached at current epoch.  Stopping.')
        break

end_time = time.time()

print('Training time: {:.2f} mins.'.format((end_time - start_time) / 60.))

# Plot the losses vs epoch here
fig = plt.figure(figsize=(8, 5))
    
plot1, = plt.plot(history['epoch'], history['loss'], c='blue', label='MAE')
plt.grid(which='both', linestyle='--')

ax = fig.gca()    
ax.set_xlabel(r'Epoch')
ax.set_ylabel(r'Loss')
plt.legend(bbox_to_anchor=(0.1, 0.0, 0.80, 1), bbox_transform=fig.transFigure, 
           loc='lower center', ncol=3, mode="expand", borderaxespad=0.)

plt.tight_layout()
plt.show()
plt.close(fig)

# Testing
model.eval()
test_loss = 0
y_pred = torch.tensor(())
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.shape[0], -1)
        output = model.forward(data)
        y_pred = torch.cat((y_pred, output), 0)
        loss = criterion(output, target)
        test_loss += batch_size * loss.item()

    test_loss /= len(test_loader.dataset)

    print('\nTest: Loss: {:.4f}'.format(test_loss))

# Predictions
y_true = test_loader.dataset.targets.view(-1, 1).detach().numpy()
y_pred = y_pred.argmax(dim=1, keepdim=True).detach().numpy()
 
# Scoring
mse = ((y_true - y_pred) ** 2).sum() / y_true.shape[0]

# Reporting the number of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: {}'.format(num_params))
