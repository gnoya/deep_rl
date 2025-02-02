import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Actor(nn.Module):
    def __init__(self, learning_rate, lr_decay, input, output):
        super(Actor, self).__init__()

        self.hidden1 = nn.Linear(input, 32)
        self.hidden2 = nn.Linear(32, 64)
        self.hidden3 = nn.Linear(64, output)
        
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.hidden3(x.view(x.size(0), -1))
        return F.softmax(x, dim=1)

    def backpropagate(self, actor_loss):
        loss = torch.cat(actor_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def learning_rate_decay(self, episode):
        lr = self.learning_rate * math.exp(- episode / self.lr_decay)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class Critic(nn.Module):
    def __init__(self, learning_rate, lr_decay, input, output):
        super(Critic, self).__init__()

        self.hidden1 = nn.Linear(input, 32)
        self.hidden2 = nn.Linear(32, 64)
        self.hidden3 = nn.Linear(64, output)
        
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.hidden3(x.view(x.size(0), -1))
        return x

    def backpropagate(self, critic_loss):
        loss = torch.cat(critic_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def learning_rate_decay(self, episode):
        lr = self.learning_rate * math.exp(- episode / self.lr_decay)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr