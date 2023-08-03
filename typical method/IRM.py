from torch.autograd import Function
import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from mlp import MLP

class IRM(MLP):
    def __init__(self, input_dim, hidden_dim, num_labels, penalty_weight):
        super(IRM, self).__init__(input_dim, hidden_dim, num_labels)
        self.penalty_weight = penalty_weight

    def penalty(self, logits, y):
        scale = torch.ones((1, self.num_labels)).to(y.device).requires_grad_()
        loss = self.criterion(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def compute_loss(self, data):
        x, y = data
        class_output = self.forward(x)
        loss = self.criterion(class_output, y)
        penalty = self.penalty(class_output, y)
        return loss + self.penalty_weight * penalty