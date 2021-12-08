# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 18:03:04 2021

@author: Anjou
"""

import torch
import numpy as np
from random import randint
from collections import deque, namedtuple

element = namedtuple('data', 'observation, action, reward, next_observation, done')

class ReplayBuffer():
    def __init__(self, 
                 capacity=10e6,
                 threshold=1000,
                 ):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.threshold = threshold
        
    def push_data(self, elem):
        if isinstance(elem, element):
            if len(self.memory) < self.capacity:
                self.memory.append(elem)
            else:
                self.memory.popleft()
                self.memory.appendleft(elem)
        else:
            raise TypeError("You are trying to push some other object than " 
                            "namedtuple('data', 'observation, action, rewrd, "
                            "next_observation, done') into the buffer")
    
    def __len__(self):
        return len(self.memory)
    
    def prepare_data(self, batch):
        size = self.__len__()
        indexes = np.random.choice(size, batch, p=[1/float(size)]*size, replace=False)
        obs = list()
        act = list()
        rew = list()
        next_obs = list()
        done = list()
        for _, idx in enumerate(indexes):
            el = self.memory[idx]
            obs.append(el[0])
            act.append(el[1])
            rew.append(el[2])
            next_obs.append(el[3])
            done.append(el[4])
        return torch.stack(obs), torch.stack(act), torch.stack(rew), torch.stack(next_obs), torch.stack(done)

    def sample(self, batch_size):
        if len(self.memory) < self.threshold:
            return None
        else:
            return self.prepare_data(batch_size)
        
    
        
if __name__=='__main__':
    mem = ReplayBuffer(100)
    print(f'Data_memory size befor is: {len(mem)}')  
    for i in range(20):
        el = element(observation=torch.rand(4,84,84),
                     action=torch.randint(0,5,(1,)),
                     reward=torch.rand(1),
                     next_observation=torch.rand(4,84,84,4), 
                     done=torch.randint(0,1,(1,)))
        mem.push_data(el)
    print(f'Data_memory size after is: {len(mem)}')    
    data = mem.sample(5)
    print(f'Data: {len(data[0])}, shape: {data[0].shape}')            
    print(f'Data_memory size after is: {len(mem)}')    
           
        
        
        
        
        