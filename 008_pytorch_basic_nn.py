#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ"""

import torch
import matplotlib.pyplot as plt

inputs = torch.rand(1, 1, 64, 64)
inputs

# category 0 and category 1
outputs = torch.rand(1, 2)
outputs

# Model
model = torch.nn.Sequential(
    # input features are the size of one image
    # outputs are how many we have when done
    # the 64 has to 'match' the final dimnension of the input
    # try changing it to another number to see errors!
    torch.nn.Linear(64, 256),
    torch.nn.Linear(256, 256),
    torch.nn.Linear(256, 2),
)

result = model(inputs)
result, result.shape

# Only LInear will not help learning, adding ReLu in next 
x = torch.range(-1, 1, 0.1)
y = torch.nn.functional.relu(x)
plt.plot(x.numpy(), y.numpy())

# Updating the model with relu
model = torch.nn.Sequential(
    torch.nn.Linear(64, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 2),
)

#Feedback by loss function (MSE), L2, Euclidean distance
results = model(inputs)
loss = torch.nn.MSELoss()(results, outputs)
loss
# Gradient tells the machine learning model, based on  the loss which direction it is away from the correct  answer
# Gradients each forward pass after computing the loss, basically zeroing out as the gradients will differ on each pass

model.zero_grad()
loss.backward()

# A constant learning rate added
learning_rate = 0.001
for parameter in model.parameters():
    parameter.data -= parameter.grad.data * learning_rate


# Running the model again
after_learning = model(inputs)
loss_after_learning = torch.nn.MSELoss()(after_learning, outputs)
loss_after_learning #smaller loss


