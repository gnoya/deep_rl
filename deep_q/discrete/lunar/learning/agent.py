import torch
import random
from q_values import QValues
import numpy as np
import copy
import os

class Agent():
    def __init__(self, policy_net, target_net, memory, strategy, gamma, num_actions, device):
        # Network to train
        self.policy_net = policy_net
        # Network to calculate ground truth
        self.target_net = target_net
        self.memory = memory
        # Epsilon Greedy strategy
        self.strategy = strategy
        self.num_actions = num_actions
        # Discount for future rewards
        self.gamma = gamma
        self.device = device
        self.steps_done = 0
        self.points_history = []

    def select_action(self, state, env):
        # Calculate epsylon-greedy threshold
        rate = self.strategy.get_exploration_rate(self.steps_done)
        self.steps_done += 1
        if random.random() > rate:
            # Return the action with the highest Q-value
            with torch.no_grad():
                state = torch.tensor([state]).float()
                if str(self.device) == 'cuda':
                    state = state.cuda()
                    out = self.policy_net(state).cpu().numpy()
                else:
                    out = self.policy_net(state).numpy()
                out = np.transpose(out, (1, 0))
                out = np.argmax(out)
        else:
            out = env.action_space.sample()

        return out

    def optimize_policy(self):
        batch_size = self.policy_net.batch_size
        if len(self.memory) < batch_size:
            return

        # Sample a random batch
        experiences = self.memory.sample(batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, next_mask, okay = self.memory.get_tensors(experiences, self.device)
        if not okay:
            return
        # Calculate Q-values of the states
        current_q_values = QValues.get_current(self.policy_net, state_batch, action_batch)
        # Calculate Q-values of the next states
        next_q_values = QValues.get_next(self.policy_net, self.target_net, next_state_batch, next_mask, self.device)
        # Calculate Q* using Bellman's equation
        target_q_values = reward_batch.float() + (self.gamma * next_q_values.float())

        # Backpropagate the policy network
        self.policy_net.backpropagate(current_q_values, target_q_values.unsqueeze(1))

    def update_target_net(self, average=False, average_tau=1):
        if not average:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(average_tau * param + (1 - average_tau) * target_param)

    def save_data(self, episode, savedir):
        last_savedir = savedir
        last_savedir += '/agent_model'
        checkpoint_file = last_savedir + '/checkpoint.pth.tar'
        points_history = []

        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            points_history = checkpoint['points_history']

            if episode == 0:
                points_history = []

        points_history += self.points_history

        data = {
            'steps': self.steps_done,
            'episode': episode,
            'optimizer': self.policy_net.optimizer.state_dict(),
            'learning_rate' : self.policy_net.learning_rate,
            'weights': self.policy_net.state_dict(),
            # 'memory' : self.memory,
            'points_history' : points_history,
        }
        self.save_weights(data, savedir)
        self.points_history = []

    def save_weights(self, data, savedir):
        savedir += '/agent_model'
        checkpoint_file = savedir + '/checkpoint.pth.tar'
        torch.save(data, checkpoint_file)