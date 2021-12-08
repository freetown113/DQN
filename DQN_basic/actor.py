# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 18:01:21 2021

@author: Anjou
"""

import torch
from envir import EnvAtari
from itertools import count
import random
from buffer import element, ReplayBuffer
from net import Network

class Actor():
    def __init__(self, 
                 env,
                 network,
                 buffer,
                 acts_size,
                 eps=0.4):
        self.eps=eps
        self.env=env
        self.net=network
        self.act_space=acts_size
        self.memory = buffer
                
    def collect(self):
        obs = self.env.reset()
        for i in count():
            threshold = random.random()
            if threshold > self.eps:
                with torch.no_grad():
                    act = self.net(obs.unsqueeze(0)).argmax(axis=-1)
            else:
                act = random.randint(0, self.act_space-1)
            next_obs, reward, done, _ = self.env.step(act)
            el = element(obs, act, reward, next_obs, done)
            self.memory.push_data(el)
            obs = next_obs
            if done:
                obs = self.env.reset()
            print(f'Buffer length os now: {len(self.memory)}')
                
                
                
if __name__=='__main__':
    def make_env():
        return EnvAtari()
    acts = make_env().action_space.n
    net = Network(4, acts)
    buffer = ReplayBuffer(100)
    actor = Actor(make_env(), net, buffer, acts)
    actor.collect()








                                