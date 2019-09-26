import random
import torch
from collections import namedtuple
import numpy as np


Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'next_state'))

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.last_position = 0

    def push(self, experience):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.last_position = self.position
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get_tensors(self, experiences, device):
        okay = True
        # "Unzip" the batch into an object
        batch = Experience(*zip(*experiences))
        # Get the values of the batch
        state_batch = torch.stack(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        next_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8).to(device)
        # Every item in the mask is zero: every item is None
        if device == 'cuda':
            tensor_to_numpy = next_mask.numpy()
        else:
            tensor_to_numpy = next_mask.cpu().numpy()

        if np.any(tensor_to_numpy):
            next_state_batch = torch.stack([s for s in batch.next_state if s is not None]).to(device)
        else:
            next_state_batch = None
            okay = False
        return state_batch, action_batch, reward_batch, next_state_batch, next_mask, okay