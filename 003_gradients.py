#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ"""

# a gradient wrt a loss function & use gradients to figure the direction to go to make loss smaller

import torch

# X + 1 = 3 

# random initialization
X = torch.rand(1, requires_grad=True)
#formula
Y = X + 1.0
Y

# Loss is how far off from 3?
def mse(Y):
    diff = 3.0 - Y
    return (diff * diff).sum() / 2

# the gradient on X tells us which direction,'off' from the right answer.Check for too high
loss = mse(Y)
loss.backward()
X.grad

# Use that gradient to solve with machine learning
learning_rate = 1e-3
# Learning loop
for i in range(0, 10000):
    Y = X + 1.0
    loss = mse(Y)
    # Backpropagation of the gradient
    loss.backward()
    # Learning, so turn off the graidents from being updated temporarily
    with torch.no_grad():
        # gradient tells in which direction its off, and you go in the opposite direction to correct the problem
        X -= learning_rate * X.grad
        # Zero out the gradients to get fresh values on each learning loop iteration
        X.grad.zero_()
# Final answer
X
# Approximate value obtained. Altering learning rate or number of iterations i.e, 'learning_rate' & number of loops in 'range'
