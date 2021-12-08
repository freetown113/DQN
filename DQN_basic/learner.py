# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 00:10:08 2021

@author: Anjou
"""

import torch
import copy


class Learner:
    def __init__(self, 
                 network,
                 buffer,
                 lr=10e-3,
                 batch=64,
                 epochs=10e6,
                 gamma=0.99
                 ):
        self.lr=lr
        self.batch=batch
        self.epochs=epochs
        self.net = network
        self.target = copy.deepcopy(network)
        self.memory = buffer
        self.loss = torch.nn.MSELoss()
        self.optim = torch.nn.optimizer.Adam(self.net.params())        
        
    def launch(self):
        
        for i in range(int(self.epochs)):
            obs, acts, reward, next_obs, done = self.memory.sample(self.batch)
            q_vals = self.net(obs)
            pred = q_vals.gather(acts, axis=-1)
            
            q_vals_next = self.target(next_obs).argmax(axis=-1)
            target = reward + self.gamma*q_vals_next
            
            error = self.loss(target, pred)
            error.backward()
            self.optim.step()
            
            
            
            
            
            
            