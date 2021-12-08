# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 20:52:20 2021

@author: Anjou
"""

from actor import Actor
from buffer import element, ReplayBuffer
from net import Network
from envir import EnvAtari
from learner import Learner
import random
import torch


if __name__=='__main__':
    def make_env():
        return EnvAtari()
    acts = make_env().action_space.n
    net = Network(4, acts)
    buffer = ReplayBuffer(1000)
    actor = Actor(make_env(), net, buffer, acts)
    for i in range(1000000):
        actor.collect()

# import torch
# conv = torch.nn.Conv1d(4,1,1)
# a = torch.randn(2, 4)
# b = torch.randn(4, 2)
# c = a.mm(b)
# e = torch.nn.functional.relu(a)
# g = torch.randn(2, 4, 1)
# d = conv(g)
# print(tuple(map(lambda t: t.is_leaf, (a, b, c, e, g, d))))
# print(tuple(map(lambda t: t.requires_grad, (a, b, c, e, g, d))))
# print(tuple(map(lambda t: t.grad_fn, (a, b, c, e, g, d))))