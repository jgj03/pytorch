#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ"""

import torch
import torchvision
import matplotlib.pyplot as plt
dir(torchvision.datasets)

# CIFAR data lookup
cifar = torchvision.datasets.CIFAR10('./var', download=True) # downloading data in var folder in workign folder
cifar[0]

# Image visualizing
fig = plt.figure(figsize=(1,1))
sub = fig.add_subplot(111)
sub.imshow(cifar[0][0])

# Image to tensor transformation
from torchvision import transforms
pipeline = transforms.Compose([
    transforms.ToTensor()
    ])
cifar_tr = torchvision.datasets.CIFAR10('./var', transform=pipeline)
# Checking
cifar_tr[0]
