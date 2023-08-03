from torch.autograd import Function
import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.criterion = nn.CrossEntropyLoss()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_labels)
        )

    def forward(self, input_data):
        feature_mapping = self.feature_extractor(input_data)
        class_output = self.label_classifier(feature_mapping)
        return class_output

    def compute_loss(self, data):
        x, y = data
        class_output = self.forward(x)
        loss = self.criterion(class_output, y)
        return loss