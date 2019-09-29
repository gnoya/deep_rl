import torch
import torch.nn as nn
import math
import sys
import time
import os
import copy
import numpy as np
from parameters import *
from agent import Agent
sys.path.append('../neural_network/')
from neural_network import *
import gym
import cv2

saved_state1 = None
saved_state2 = None
saved_state3 = None
saved_state4 = None

def initialize():
    # Define PyTorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Variable env contains the environment class (python game)
    env = gym.envs.make(env_name)
    # Define policy network
    actor = Actor(actor_learning_rate, actor_lr_decay, 80, 4).to(device).float()
    critic = Critic(critic_learning_rate, critic_lr_decay, 80, 1).to(device).float()
    agent = Agent(actor, critic, gamma, device)

    return env, agent

def preprocess_state(state):
    global saved_state1, saved_state2, saved_state3, saved_state4

    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state[50:,:], (80, 80)) 

    if(saved_state1 is not None):
        saved_state4 = saved_state3
        saved_state3 = saved_state2
        saved_state2 = saved_state1
        saved_state1 = state
    else:
        saved_state1 = state
        saved_state2 = state
        saved_state3 = state
        saved_state4 = state

    state = np.dstack((saved_state1, saved_state2, saved_state3, saved_state4))
    state = np.transpose(state, (2, 0, 1))
        
    return state

def train():
    env, agent = initialize()
    for episode in range(num_episodes):

        state = env.reset()
        state = preprocess_state(state)
        done = False
        env.render()
        steps = 0
        I = 1
        while(not done):
            steps += 1
            env.render()

            action = agent.select_action(state, env)
            next_state, reward, done, _ = env.step(action)
            reward = reward if reward != 0 else 0
            reward = reward if not done else -1
            next_state = preprocess_state(next_state)


            value = agent.estimate_value(state)
            print('Value: {}'.format(value.item()))
            next_value = agent.estimate_value(next_state) if not done else 0
            target = (reward + gamma * next_value)
            advantage = target - value
            print('Adv: {}'.format(advantage.item()))
            agent.optimize(I * advantage)

            state = next_state
            I *= 1

        if actor_lr_decay_active:
            agent.actor.learning_rate_decay(episode)
        if critic_lr_decay_active:
            agent.critic.learning_rate_decay(episode)

        print('Steps: {}'.format(steps))
        print('Episode: {}'.format(episode))
        print('Actor learning rate: {}'.format(agent.actor.optimizer.param_groups[0]['lr']))
        print('Critic learning rate: {}\n'.format(agent.critic.optimizer.param_groups[0]['lr']))
        

if __name__ == '__main__':
    train()
