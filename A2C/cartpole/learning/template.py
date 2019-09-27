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
    actor = Actor(batch_size, learning_rate, 4, 2).to(device).float()
    critic = Critic(batch_size, learning_rate, 4, 2).to(device).float()

    agent = Agent(actor, critic, gamma, device)

    return env, agent

def train():
    env, agent = initialize()
    k = np.array([])
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

            state = next_state

        if(len(k) < 100):
            k = np.append(k, steps)
        else:
            k = k[1:]
            k = np.append(k, steps)
        
        if np.mean(k) > 195.0 and len(k) == 100:
            print('Solved in {} episodes'.format(episode - 100))
            time.sleep(25)

        agent.optimize_policy()

        if lr_decay_active:
            agent.policy_net.learning_rate_decay(episode, lr_decay)

        print('Steps: {}'.format(steps))
        print('Episode: {}'.format(episode))
        print('Mean of 100: {}'.format(np.mean(k)))
        print('Learning_rate: {}\n'.format(agent.policy_net.optimizer.param_groups[0]['lr']))

if __name__ == '__main__':
    train()
