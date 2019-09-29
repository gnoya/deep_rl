import torch
import torch.nn as nn
import math
from argparse import ArgumentParser
import sys
import time
import os
import copy
import numpy as np
from parameters import *
from replay_memory import *
from agent import Agent
from epsilon_greedy import EpsilonGreedy
from q_values import QValues
sys.path.append('../neural_network/')
from neural_network import *
import gym

savedir = ''

def initialize(args):
    global savedir

    savedir = '../instances'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    savedir = '../instances/{}'.format(args.save_instance)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        os.makedirs(savedir + '/agent_model')

    # Define PyTorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Variable env contains the environment class (python game)
    env = gym.envs.make("CartPole-v1")

    # Define policy and target networks
    policy_net = DQN(batch_size, learning_rate, 4, 2).to(device).float()
    target_net = DQN(batch_size, learning_rate, 4, 2).to(device).float()
    # Copy the weights
    target_net.load_state_dict(policy_net.state_dict())
    # Do not backpropagate target network
    target_net.eval()

    memory = ReplayMemory(memory_size)
    strategy = EpsilonGreedy(eps_start, eps_end, eps_decay)
    agent = Agent(policy_net, target_net, memory, strategy, gamma, 2, device)

    return env, agent

def train(args):
    global savedir
    env, agent = initialize(args)
    k = np.array([])
    # Loop through a lot of domino games
    for episode in range(num_episodes):
        start = time.time()
        state = env.reset()
        done = False
        env.render()
        steps = 0
        # Loop through one domino game
        while(not done):
            steps += 1
            env.render()
            # Select and perform an action
            action = agent.select_action(state, env)

            # state, action, reward and next_state are PyTorch tensors
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -reward
            reward = 0.1 if reward == 1.0 else reward
            state = torch.tensor(state)
            action = torch.tensor([[action]])
            reward = torch.tensor([reward])
            next_state_ = torch.tensor(next_state)
            # Store the experience in memory
            agent.memory.push(Experience(state, action, reward, next_state_))

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.optimize_policy()

        agent.points_history.append(steps)

        if(len(k) < 100):
            k = np.append(k, steps)
        else:
            k = k[1:]
            k = np.append(k, steps)
        
        if np.mean(k) > 195.0 and len(k) == 100:
            print('Solved in {} episodes'.format(episode - 100))
            time.sleep(25)
        
        # Update the target network, copying all weights and biases in the target network
        if agent.steps_done % target_network_freq == 0:
            agent.update_target_net()

        # Save agents
        if episode % saving_model_freq == 0:
            agent.save_data(episode, savedir)

        # learing rate decay
        if lr_decay_active:
            agent.policy_net.learning_rate_decay(episode, lr_decay)

        print('Steps: {}'.format(steps))
        print('Episode: {} | Time: {}'.format(episode, round(time.time() - start, 2)))
        #print('Steps done: {}'.format(agent.steps_done))
        print('Eps threshold: {}'.format(agent.strategy.get_exploration_rate(agent.steps_done)))
        print('Mean of 100: {}\n'.format(np.mean(k)))
        print('Learning_rate: {}\n'.format(agent.policy_net.optimizer.param_groups[0]['lr']))

    print('Done')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--resume')
    parser.add_argument('--save_instance', required=True)
    train(parser.parse_args())
