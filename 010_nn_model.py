#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ"""


import torch

inputs = torch.rand(1, 1, 64, 64)
outputs = torch.rand(1, 2)

learning_steps = []

for number_of_parameters in range(256, 1, -1):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_one = torch.nn.Linear(64, number_of_parameters)
            self.activation_one = torch.nn.ReLU()
            self.layer_two = torch.nn.Linear(number_of_parameters, number_of_parameters)
            self.activation_two = torch.nn.ReLU()
            self.shape_outputs = torch.nn.Linear(number_of_parameters * 64, 2)

        def forward(self, inputs):
            buffer = self.layer_one(inputs)
            buffer = self.activation_one(buffer)
            buffer = self.layer_two(buffer)
            buffer = self.activation_two(buffer)
            buffer = buffer.flatten(start_dim=1)
            return self.shape_outputs(buffer)

    model = Model()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i in range(10000):
        optimizer.zero_grad()
        results = model(inputs)
        loss = loss_function(results, outputs)
        loss.backward()
        optimizer.step()
        gradients = 0.0
        for parameter in model.parameters():
            gradients += parameter.grad.data.sum()
        if abs(gradients) <= 0.0001:
            learning_steps.append((number_of_parameters, i, results))
            break
learning_steps

# Visulaizing
import matplotlib.pyplot as plt
plt.style.use('ggplot')
learning_steps = [step[1] for step in learning_steps]
plt.plot(learning_steps)

