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

def initialize():
    # Define PyTorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Variable env contains the environment class (python game)
    env = gym.envs.make(env_name)

    # Define policy network
    policy_net = DQN(batch_size, learning_rate, 4, 2).to(device).float()

    agent = Agent(policy_net, gamma, device)

    return env, agent

def train():
    env, agent = initialize()
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        env.render()
        steps = 0

        while(not done):
            steps += 1
            env.render()

            action = agent.select_action(state, env)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -reward

            agent.rewards.append(reward)

            state = next_state

        agent.optimize_policy()

        if lr_decay_active:
            agent.policy_net.learning_rate_decay(episode, lr_decay)

        print('Steps: {}'.format(steps))
        print('Episode: {}'.format(episode))
        print('Learning_rate: {}\n'.format(agent.policy_net.optimizer.param_groups[0]['lr']))

if __name__ == '__main__':
    train()
