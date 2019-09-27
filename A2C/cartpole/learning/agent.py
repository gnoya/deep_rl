import torch
import random
import numpy as np
import copy
import os
from torch.distributions import Categorical

class Agent():
    def __init__(self, actor, critic, gamma, device):
        # Network to train
        self.actor = actor
        self.critic = critic
        
        # Discount for future rewards
        self.gamma = gamma
        self.device = device
        self.steps_done = 0

    def select_action(self, state, env):
        self.steps_done += 1
        state = torch.tensor([state]).float()
        if str(self.device) == 'cuda':
            state = state.cuda()
            out = self.actor(state)
        else:
            out = self.actor(state)
        m = Categorical(out)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def optimize_policy(self):
        R = 0
        policy_loss = []
        returns = []

        # Monte carlo
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        # Normalize the rewards
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Calculate loss for every step in Monte Carlo
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        # Backpropagate the policy network
        self.actor.backpropagate(policy_loss)
        