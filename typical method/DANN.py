from torch.autograd import Function
import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lamda
        return output, None

class DANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_labels, num_domains, lamda):
        super(DANN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_domains = num_domains
        self.lamda = lamda
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

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_domains)
        )

    def forward(self, input_data):
        feature_mapping = self.feature_extractor(input_data)
        reverse_feature = ReverseLayerF.apply(feature_mapping, self.lamda)
        class_output = self.label_classifier(feature_mapping)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output

    def compute_loss(self, source_data, target_data):
        inputs, class_labels, domain_source_labels = source_data
        pred_class_label, pred_domain_label = self.forward(inputs)
        class_loss = self.criterion(pred_class_label, class_labels.long())
        domain_source_loss = self.criterion(pred_domain_label, domain_source_labels)

        inputs, domain_target_labels = target_data
        _, pred_domain_label = self.forward(inputs)
        domain_target_loss = self.criterion(pred_domain_label, domain_target_labels)

        domain_loss = domain_source_loss + domain_target_loss
        return class_loss, domain_loss
    