import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DQN(nn.Module):
    def __init__(self, batch_size, learning_rate, input, output):
        super(DQN, self).__init__()

        self.hidden1 = nn.Linear(input, 8)
        self.hidden2 = nn.Linear(8, 64)
        self.hidden3 = nn.Linear(64, 32)
        self.hidden4 = nn.Linear(32, output)
        self.last_learing_rate_reset = 0
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))

        return self.hidden4(x.view(x.size(0), -1))

    def backpropagate(self, y_pred, y):
        loss = F.smooth_l1_loss(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def learning_rate_decay(self, episode, decay):
        lr = self.learning_rate * math.exp(- (episode - self.last_learing_rate_reset) / decay)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr