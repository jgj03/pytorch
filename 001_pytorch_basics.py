#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ"""

import torch
torch.__version__

# https://pytorch.org/docs/stable/torch.html#creation-ops
e = torch.empty(2, 2)
e

r = torch.rand(2, 2)
r

z = torch.zeros(2, 2)
z

c = torch.full((2, 2), 3)
c 

l = torch.tensor([[1, 2], [3, 4]])
l

import numpy
n = numpy.linspace(0, 5, 5)
n

nn = torch.tensor(n)
nn

nn.numpy()

nn.shape

nn[1:], nn[0]

s = torch.ones(3, 3, dtype=torch.float)
s

# https://pytorch.org/docs/stable/torch.html#math-operations
eye = torch.eye(3, 3)
eye + torch.zeros(3, 3)

# subtraction
eye - torch.ones(3, 3)

# broadcast multiplication of a constant
eye * 3

# division...
eye / 3

# element wise tensor multiplication
eye * torch.full((3,3), 4)

# a dot product operator in python
x = torch.rand(3, 4)
y = torch.rand(4, 3)
x @ y

# A handy machine learning component operations like getting the index of the maximum value
torch.tensor([1 , 2, 5, 3, 0]).argmax()
