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
        self.saved_log_probs = []
        self.saved_values = []

    def select_action(self, state, env):
        self.steps_done += 1
        state = torch.tensor([state]).float()
        if str(self.device) == 'cuda':
            state = state.cuda()
            out = self.actor(state)
        else:
            out = self.actor(state)

        print('Prob: {}'.format(out.detach()))
        m = Categorical(out)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def estimate_value(self, state):
        state = torch.tensor([state]).float()
        if str(self.device) == 'cuda':
            state = state.cuda()
            out = self.critic(state)
        else:
            out = self.critic(state)
        self.saved_values.append(out.squeeze(0))
        return out.squeeze(0)

    def optimize(self, advantage):
        actor_loss = []
        critic_loss = []

        actor_loss.append(-self.saved_log_probs[0] * advantage.detach())
        critic_loss.append(advantage ** 2)
        
        # Backpropagate networks
        self.actor.backpropagate(actor_loss)
        self.critic.backpropagate(critic_loss)

        self.saved_log_probs = []
        self.saved_values = []