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
    print(env.unwrapped.get_action_meanings())
    assert False
    # Define policy network
    actor = Actor(actor_learning_rate, actor_lr_decay, 4, 2).to(device).float()
    critic = Critic(critic_learning_rate, critic_lr_decay, 4, 1).to(device).float()

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
        I = 1
        while(not done):
            steps += 1
            env.render()

            action = agent.select_action(state, env)
            next_state, reward, done, _ = env.step(action)
            reward = 1 if not done else -reward
            if(steps == 500):
                reward = 2.0
                print('Reward 500')

            value = agent.estimate_value(state)
            next_value = agent.estimate_value(next_state) if not done else 0
            target = (reward + gamma * next_value)
            advantage = target - value
            agent.optimize(I * advantage)

            state = next_state
            I *= 1

        if(len(k) < 100):
            k = np.append(k, steps)
        else:
            k = k[1:]
            k = np.append(k, steps)
        
        if np.mean(k) > 195.0 and len(k) == 100:
            print('Solved in {} episodes'.format(episode - 100))
            time.sleep(25)

        if actor_lr_decay_active:
            agent.actor.learning_rate_decay(episode)
        if critic_lr_decay_active:
            agent.critic.learning_rate_decay(episode)

        print('Steps: {}'.format(steps))
        print('Episode: {}'.format(episode))
        print('Mean of 100: {}'.format(np.mean(k)))
        print('Actor learning rate: {}'.format(agent.actor.optimizer.param_groups[0]['lr']))
        print('Critic learning rate: {}\n'.format(agent.critic.optimizer.param_groups[0]['lr']))
        

if __name__ == '__main__':
    train()
