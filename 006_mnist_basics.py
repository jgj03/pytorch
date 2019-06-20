#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ"""

import torchvision
import torch

#MNIST data
mnist = torchvision.datasets.MNIST('./var', download=True, transform=torchvision.transforms.ToTensor())

mnist[0]

# Tensor of batch averages
batches = torch.utils.data.DataLoader(mnist, 
    batch_size=32)

batch_averages = torch.Tensor([
    batch[0].mean() for batch in batches
])

batch_averages.mean()


# Maching learning with batch training
all_images = torch.cat([
    image for image, label in mnist
])

all_images.shape, all_images.mean()
