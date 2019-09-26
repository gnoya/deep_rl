import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, learning_rate,batch_size,  num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.batch_size = batch_size
        self.learning_rate = learning_rate # !
        self.last_learing_rate_reset = 0 # !
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def backpropagate(self, y_pred, y):
        loss = F.smooth_l1_loss(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learning_rate_decay(self,episode,decay):
        lr = self.learning_rate * math.exp(- (episode - self.last_learing_rate_reset) / decay)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_learning_rate(self,episode,reset_scale):
        # alpha = alpha' + (alpha - alpha') * scale; where alpha is the initial learning_rate and alpha' is the final learning_rate
        self.learning_rate = self.optimizer.param_groups[0]['lr'] + ((self.learning_rate - self.optimizer.param_groups[0]['lr']) * reset_scale)
        self.last_learing_rate_reset = episode

def resnet20(batch_size,learning_rate):
    return ResNet(BasicBlock, [3, 3, 3],learning_rate,batch_size)

def resnet32(batch_size,learning_rate):
    return ResNet(BasicBlock, [5, 5, 5],learning_rate,batch_size)

def resnet44(batch_size,learning_rate):
    return ResNet(BasicBlock, [7, 7, 7],learning_rate,batch_size)

def resnet56(batch_size,learning_rate):
    return ResNet(BasicBlock, [9, 9, 9],learning_rate,batch_size)

def resnet110(batch_size,learning_rate):
    return ResNet(BasicBlock, [18, 18, 18],learning_rate,batch_size)

def resnet1202(batch_size,learning_rate):
    return ResNet(BasicBlock, [200, 200, 200],learning_rate,batch_size)
