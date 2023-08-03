from torch.autograd import Function
import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as n
from mlp import MLP

class REx(MLP):
    def __init__(self, input_dim, hidden_dim, num_labels, variance_weight):
        super(REx, self).__init__(input_dim, hidden_dim, num_labels)
        self.variance_weight = variance_weight
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def compute_loss(self, data):
        x, y = data
        class_output = self.forward(x)
        loss = self.criterion(class_output, y)
        loss_mean = torch.mean(loss)
        loss_var = torch.var(loss)
        return loss_mean + self.variance_weight * loss_var

