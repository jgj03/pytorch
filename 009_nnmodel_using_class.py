#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ"""

import torch

# Model is made with linear and relu
inputs = torch.rand(1, 1, 64, 64)
outputs = torch.rand(1, 2)

# Reusable model
class Model(torch.nn.Module):

    def __init__(self):
        '''
        The constructor is the place to set up each of the layers
        and activations.
        '''

        super().__init__()
        self.layer_one = torch.nn.Linear(64, 256)
        self.activation_one = torch.nn.ReLU()
        self.layer_two = torch.nn.Linear(256, 256)
        self.activation_two = torch.nn.ReLU()
        self.shape_outputs = torch.nn.Linear(16384, 2)

    def forward(self, inputs):
        buffer = self.layer_one(inputs)
        buffer = self.activation_one(buffer)
        buffer = self.layer_two(buffer)
        buffer = self.activation_two(buffer)
        buffer = buffer.flatten(start_dim=1)
        return self.shape_outputs(buffer)

# Rnnign the model
model = Model()
test_results = model(inputs)
test_results

# Learning loop with a built in optimizer. Stop learning when gradients 'vanish' 
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# check learning done
for i in range(10000):
    optimizer.zero_grad()
    results = model(inputs)
    loss = loss_function(results, outputs)
    loss.backward()
    optimizer.step()
    # Vanishing gradients
    gradients = 0.0
    for parameter in model.parameters():
        gradients += parameter.grad.data.sum()
    if abs(gradients) <= 0.0001:
        print(gradients)
        print('gradient vanished at iteration {0}'.format(i))
        break

# Answer
model(inputs), outputs
# Need to check for overfitting as this is random data and stil fitting. 
