#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" @author: JGJ"""

# Visualization and NN checking
#import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
#import math

# Building a graph with neural network X
dense = nx.Graph()
inputs = {i: (0, i) for i in range(0, 5)}
activations = {i+100: (1, i) for i in range(0, 5)}
outputs= {i+1000: (2, i) for i in range(0, 2)}
all = {**inputs, **activations, **outputs}

# FCN: , every input talks to every activation. classic NN
for input in inputs:
    for activation in activations:
        dense.add_edge(input, activation)
for activation in activations:
    for output in outputs:
        dense.add_edge(activation, output)
nx.draw_networkx_nodes(dense, all, 
    nodelist=all.keys(), node_color='b')
nx.draw_networkx_edges(dense, all, edge_color='w')
axes = plt.axis('off')
