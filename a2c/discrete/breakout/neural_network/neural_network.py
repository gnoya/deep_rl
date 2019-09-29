import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Actor(nn.Module):
    def __init__(self, learning_rate, lr_decay, input, output):
        super(Actor, self).__init__()
        assert input % 4 == 0
        self.input = input
        self.output = output
        
        self.conv1 = torch.nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(16 * int(self.input/4) * int(self.input/4), 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, self.output)
        
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * int(self.input/4) * int(self.input/4))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.softmax(x.view(x.size(0), -1), dim=1)

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
        
        assert input % 4 == 0
        self.input = input
        self.output = output
        
        self.conv1 = torch.nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(16 * int(self.input/4) * int(self.input/4), 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, self.output)
        
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * int(self.input/4) * int(self.input/4))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        
        return x.view(x.size(0), -1)

    def backpropagate(self, critic_loss):
        loss = torch.cat(critic_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def learning_rate_decay(self, episode):
        lr = self.learning_rate * math.exp(- episode / self.lr_decay)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr