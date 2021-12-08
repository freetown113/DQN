import gym
import torch
from itertools import count
from random import randint
from PIL import Image
import numpy as np
import os
from gym.envs.atari.atari_env import AtariEnv

class EnvAtari(AtariEnv):
    def __init__(self, game='pong', 
                         obs_type='image', 
                         frameskip=1,
                         colour=False,
                         repeat_action_probability=0.,
                         frame_stack=4):
        super().__init__(game=game, 
                         obs_type=obs_type, 
                         frameskip=frameskip, 
                         repeat_action_probability=repeat_action_probability)
        self.size = (84, 84)
        self.colour = colour
        self.framestack = frame_stack
        if self.colour == True:
            channels = 3
        elif self.colour == False:
            channels = 1
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.size[0], self.size[1], channels),
            dtype=np.uint8
            )
        
    def get_image(self):
        img = Image.fromarray(super()._get_image())
        resized = img.resize(self.size)
        if self.colour == True:
            return np.asarray(resized)
        elif self.colour == False:
            return (np.asarray(resized)[...,:1]/255.).astype('float32')
        
    def reset(self):
        super().reset()
        return torch.from_numpy(np.stack([self.get_image() for i in range(self.framestack)], axis=-1)).reshape(-1, *self.size) #self.get_image())
    
    def step(self, action):
        reward = 0.0
        observ = list()
        if isinstance(self.framestack, int):
            num_steps = self.framestack
        else:
            num_steps = self.np_random.randint(self.framestack[0], self.framestack[1])
        for _ in range(num_steps):
            obs, rew, done, info = super().step(action)
            reward += rew
            observ.append(self.get_image())

        ob = np.stack(observ, axis=-1).reshape(-1, *self.size)
        return torch.from_numpy(ob), torch.tensor([reward]), torch.tensor([done]), _
        
    def save_images(self, id):
        print(f'observation: {ob.shape}, {type(ob[:1,...].squeeze(0))}')
        img = Image.fromarray(ob[:1,...].squeeze(0), mode='RGB')
        #print(f'image: {img}')
        img.save("C:/Users/Anjou/Pictures/test/"+str(id)+'.jpg',"JPEG")
                

if __name__=="__main__":
    env = EnvAtari()
    act_space = env.action_space
    obs_space = env.observation_space
    print(f'action_space: {act_space}, obs_space: {obs_space}')
    state = env.reset()
    print(f'State = {state.shape, type(state)}')
    for i in range(5):
        action = randint(0,5)
        ob, reward, done, info = env.step(action)
        print(f'obs: {ob.shape}')
        print(f'rew: {reward}')
        print(f'done: {done}')
        print(f'info: {info}')
        
        
        