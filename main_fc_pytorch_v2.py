#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 21:29:51 2025

@author: farismismar
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc, precision_recall_curve

import matplotlib.pyplot as plt
import datetime

import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

data_folder = './dataset'
feature_name = 'feature'
seed = 42  # for reproducibility
prefer_gpu = True
random_state = np.random.RandomState(seed=seed)
torch.manual_seed(seed)

epoch_count = 10
learning_rate = 1e-4
batch_size = 64
depth = 5
width = 30

use_cuda = torch.cuda.is_available() and prefer_gpu
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.enabled = use_cuda

df = pd.read_csv(os.path.join(data_folder, 'dataset.csv'))
y = df[feature_name].values
X = df.drop(feature_name, axis=1)

feature_names = X.columns

X_train, X_cv, y_train, y_cv = train_test_split(X, y, train_size=0.8, random_state=random_state)

# Construct the DNN 
input_dimension = X_train.shape[1]

# Initialize list to store layers
layers = []

# First layer
layers.append(nn.Linear(input_dimension, width, bias=True))
layers.append(nn.ReLU())

# Add hidden layers based on depth
for _ in range(depth - 1):
    layers.append(nn.Linear(width, width, bias=True))
    layers.append(nn.ReLU())

# Output layer
layers.append(nn.Linear(width, 1))
layers.append(nn.Sigmoid())

# Create Sequential model
model = nn.Sequential(*layers).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

# Training
history = {'epoch': [], 'training_loss': [], 'training_f1': [], 'cv_f1': []}

# Convert NumPy arrays to PyTorch tensors
X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
X_cv = torch.from_numpy(X_cv).float().to(device)
y_cv = torch.from_numpy(y_cv).float().to(device)

# Create TensorDataset and DataLoader for mini-batches
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

cv_dataset = TensorDataset(X_cv, y_cv)
cv_loader = DataLoader(cv_dataset, batch_size=1, shuffle=True)

y_train_pred = []
y_train_true = []

start_time = time.time()
for epoch in np.arange(epoch_count):
    model.train()
    epoch_loss = 0

    # Iterate on batches for stochastic behavior
    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        # batch_X = batch_X.view(batch_X.shape[0], -1) # reshape

        optimizer.zero_grad()  # minimize
        outputs = model.forward(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()  # backward propagation
        optimizer.step()  # update optimizer
        epoch_loss += batch_size * loss.item()
    
        batch_y_pred = (outputs >= 0.5).float()
        y_train_pred.extend(batch_y_pred.cpu().numpy())
        y_train_true.extend(batch_y.cpu().numpy())
        
        train_f1_performance = f1_score(batch_y.cpu().numpy(), batch_y_pred.cpu().numpy())
               
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f},\tF1: {:.4f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item(), train_f1_performance))
        
    # Performance
    train_f1_perf_i = f1_score(y_train_pred, y_train_true)
    
    model.eval()
    with torch.no_grad():
        y_proba = model(X_cv)
        prediction = (y_proba >= 0.5).float()
        cv_f1_perf_i = f1_score(y_cv.cpu().numpy(), prediction.cpu().numpy())
    
    history['epoch'].append(epoch)
    history['training_loss'].append(epoch_loss / len(train_loader.dataset))
    history['training_f1'].append(train_f1_perf_i)
    history['cv_f1'].append(cv_f1_perf_i)

end_time = time.time()

print('Training time: {:.2f} mins.'.format((end_time - start_time) / 60.))

fpr_dnn, tpr_dnn, threshold = roc_curve(y_cv, y_proba)
roc_auc_dnn = auc(fpr_dnn, tpr_dnn)
y_pred_dnn = prediction.cpu().numpy().astype(int)
precision_dnn, recall_dnn, _ = precision_recall_curve(y_cv, y_pred_dnn)
confusion_matrix(y_cv, y_pred_dnn)
validation_accuracy = accuracy_score(y_cv, y_pred_dnn)
validation_f1 = f1_score(y_cv, y_pred_dnn)
print(validation_accuracy, validation_f1)

# ROC Plot
fig = plt.figure(figsize=(8, 5))
plt.plot(fpr_dnn, tpr_dnn, 'b', label = f'DNN AUC = {roc_auc_dnn:.4f}')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close(fig)

# PR Plot
fig = plt.figure(figsize=(8,5))
plt.plot(precision_dnn, recall_dnn, 'b', label="DNN", linewidth=2)
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close(fig)
