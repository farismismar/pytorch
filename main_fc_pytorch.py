# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 08:56:31 2020

@author: farismismar
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np
import matplotlib.pyplot as plt

import time

n_epochs = 100
learning_rate = 0.01
momentum = 0.5
batch_size = 16
prefer_gpu = True

accuracy_threshold = 0.99

input_dim = 28 ** 2 # MNIST input size
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

# Construct the DNN 
model = torch.nn.Sequential(
         nn.Linear(input_dim, hidden_1_dim, bias=True),
         nn.ReLU(),
         nn.Linear(hidden_1_dim, hidden_2_dim, bias=True),
         nn.ReLU(),
         nn.Linear(hidden_1_dim, output_dim, bias=True),
         nn.Softmax(dim=1)
        ).to(device)

# Initialize the weights for better grad descent behavior
alpha = 1. / np.sqrt(hidden_1_dim)
nn.init.uniform_(model[0].weight, a=-alpha, b=alpha)
nn.init.uniform_(model[0].bias, a=-alpha, b=alpha)
alpha = 1. / np.sqrt(hidden_2_dim)
nn.init.uniform_(model[2].weight, a=-alpha, b=alpha)
nn.init.uniform_(model[2].bias, a=-alpha, b=alpha)
alpha = 1. / np.sqrt(output_dim)
nn.init.uniform_(model[4].weight, a=-alpha, b=alpha)
nn.init.uniform_(model[4].bias, a=-alpha, b=alpha)

# GD with adaptive moments
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Training
history = {'epoch': [], 'loss': [], 'accuracy': []}

start_time = time.time()
for epoch in np.arange(n_epochs):
    model.train()
    epoch_loss = 0
    correct_train = total_train = 0 
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.shape[0], -1) # reshape

        optimizer.zero_grad() # minimize
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward() # backward propagation
        optimizer.step() # update optimizer
        
        epoch_loss += batch_size * loss.item() # since loss reduction is by mean
        
        # accuracy
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max probability
        correct_train += pred.eq(target.view_as(pred)).sum().item()
        total_train += target.nelement()
        train_accuracy = correct_train / total_train
        
        if batch_idx % 50 == 0:
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
plt.clf()
fig = plt.figure(figsize=(8, 5))
plt.plot(history['epoch'], history['loss'], marker='o', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Performance')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close(fig)

plt.clf()
fig = plt.figure(figsize=(8, 5))
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
        data = data.view(data.shape[0], -1)
        output = model(data)
        loss = F.cross_entropy(output, target)
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
