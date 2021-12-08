# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 00:12:50 2021

@author: Anjou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(in_shape, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.flatten = nn.Linear(32*9*9, 256)
        self.linear = nn.Linear(256, out_shape)
        
    def forward(self, x):
        #print(f'shape0: {x.shape}')
        x = F.relu(self.conv1(x))
        #print(f'shape1: {x.shape}')
        x = F.relu(self.conv2(x))
        #print(f'shape2: {x.shape}')
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.flatten(x))
        #print(f'shape3: {x.shape}')
        x = self.linear(x)
        #print(f'shape4: {x.shape}')
        return x
        
        
if __name__=='__main__':
    net = Network(4, 5)
    input = torch.rand(10, 4, 84, 84)
    print(f'Tensor shape: {input.shape}')
    out = net(input)