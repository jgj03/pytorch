#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ"""

# Dataset, Dataloader, and transforms & use pandas and CSV data to create a dataset.

import torch
import pandas
from torch.utils.data import Dataset

class IrisDataset(Dataset):

    def __init__(self):
        '''Load up the data.
        '''
        self.data = pandas.read_csv('home/.../Iris.csv')

    def __len__(self):
        '''How much data do we have?
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''Grab one data sample
        
        Arguments:
            idx {int} -- data at this position.
        '''
        return self.data.iloc[idx]

# Iris dataset loading
iris = IrisDataset()
len(iris), iris[0]

# Tuple to tensor transform

class TensorIrisDataset(IrisDataset):
    def __getitem__(self, idx):
        '''Get a single sample that is 
        {values:, label:}
        '''
        sample = super().__getitem__(idx)
        return {
            'tensor': torch.Tensor(
                [sample.SepalLengthCm,
                sample.SepalWidthCm,
                sample.PetalLengthCm,
                sample.PetalWidthCm]
            ),
            'label': sample.Species
        }

# Output
tensors = TensorIrisDataset()
len(tensors), tensors[0]

# Training in batches
from torch.utils.data import DataLoader

loader = DataLoader(tensors, batch_size=16, shuffle=True)
for batch in loader:
    print(batch)

# Parallel possibilities
parallel_loader = DataLoader(tensors, 
    batch_size=16, shuffle=True, num_workers=4)
for batch in parallel_loader:
    print(batch)
