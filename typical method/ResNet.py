from torch.autograd import Function
import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels):
        super(ResNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.criterion = nn.CrossEntropyLoss()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(input_dim, hidden_dim))
        for i in range(3):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
        self.lins.append(nn.Linear(hidden_dim, num_labels))

    def forward(self, input_data):
        x = self.lins[0](input_data)
        x = F.relu(x, inplace=True)
        for i, lin in enumerate(self.lins[1:-1]):
            x_ = lin(x)
            x_ = F.relu(x_, inplace=True)
            x = x_ + x
        x = self.lins[-1](x)
        return x

    def compute_loss(self, data):
        x, y = data
        class_output = self.forward(x)
        loss = self.criterion(class_output, y)
        return loss

