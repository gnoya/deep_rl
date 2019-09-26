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
from Resnet import *
import gym

savedir = ''


def initialize(args):
    global savedir
    agents = []
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
    env = gym.envs.make(env_name)

    for i in range(3):
        # Define policy and target networks
        policy_net = resnet20(batch_size, learning_rate, 24, outputs + 1).to(device).float()
        target_net = resnet20(batch_size, learning_rate, 24, outputs + 1).to(device).float()
        # Copy the weights
        target_net.load_state_dict(policy_net.state_dict())
        # Do not backpropagate target network
        target_net.eval()

        memory = ReplayMemory(memory_size)
        strategy = EpsilonGreedy(eps_start, eps_end, eps_decay)
        agent = Agent(policy_net, target_net, memory, strategy, gamma, 2, device)
        agents.append(agent)

    return env, agents

def train(args):
    global savedir
    env, agents = initialize(args)
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
            actions = []
            original_actions = []
            for agent in agents:
                out = agent.select_action(state, env)
                a = (out - outputs / 2) / (outputs / 2)
                actions.append(a)
                original_actions.append(out)
            
            # state, action, reward and next_state are PyTorch tensors
            next_state, reward, done, _ = env.step(actions)

            if(steps > 200):
                done = True
                reward -= 0
                print('out')

            state = torch.tensor(state)
            reward = torch.tensor([float(reward)])
            next_state_ = torch.tensor(next_state)

            # Store the experience in memory
            for agent, action in zip(agents, original_actions):
                act = torch.tensor([[action]])
                agent.memory.push(Experience(state, act, reward, next_state_))

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            for agent in agents:
                agent.optimize_policy()
                agent.update_target_net(average=target_update_average_active, tau=target_update_average_tau)

        # Update the target network, copying all weights and biases in the target network
        # for agent in agents:
        #     # if agent.steps_done % target_network_freq == 0:
        #     agent.update_target_net(average=target_update_average_active, tau=target_update_average_tau)

        # # Save agents
        # if episode % saving_model_freq == 0:
        #     agent.save_data(episode, savedir)

        # learing rate decay
        if lr_decay_active:
            for agent in agents:
                agent.policy_net.learning_rate_decay(episode, lr_decay)
        print(steps)
        print('Episode: {} | Time: {}'.format(episode, round(time.time() - start, 2)))
        #print('Steps done: {}'.format(agent.steps_done))
        print('Eps threshold: {}'.format(agents[0].strategy.get_exploration_rate(agents[0].steps_done)))
        print('Learning_rate: {}\n'.format(agents[0].policy_net.optimizer.param_groups[0]['lr']))

    print('Done')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--resume')
    parser.add_argument('--save_instance', required=True)
    train(parser.parse_args())
