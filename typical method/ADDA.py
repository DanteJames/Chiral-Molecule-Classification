from torch.autograd import Function
import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class ADDA:
    def __init__(self, input_dim, hidden_dim, num_labels, num_domains, lamda):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_domains = num_domains
        self.lamda = lamda
        self.criterion = nn.CrossEntropyLoss()

        self.srcMapper = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, input_dim)
        )

        self.tgtMapper = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, input_dim)
        )

        self.Classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, num_labels)
        )

        self.Discriminator = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_domains)
        )

    def pretrain_forward(self, input_data):
        src_mapping = self.srcMapper(input_data)
        src_class = self.Classifier(src_mapping)
        return src_class

    def pretrain_loss(self, train_data):
        x, y = train_data
        src_class = self.pretrain_forward(x)
        loss = self.criterion(src_class, y.long())
        return loss

    def discriminator_loss(self, src_x, tgt_x):
        src_mapping = self.srcMapper(src_x)
        tgt_mapping = self.tgtMapper(tgt_x)
        batch_size = src_mapping.size(0)
        alpha = torch.rand(batch_size, 1).to(src_mapping.device)
        alpha = alpha.expand(src_mapping.size())
        interpolates = alpha * src_mapping + (1 - alpha) * tgt_mapping
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.Discriminator(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(interpolates.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        pred_tgt, pred_src = self.Discriminator(tgt_mapping), self.Discriminator(src_mapping)
        loss_discriminator = pred_tgt.mean() - pred_src.mean() + self.lamda * gradient_penalty
        return loss_discriminator

    def tgt_loss(self, tgt_x):
        tgt_mapping = self.tgtMapper(tgt_x)
        pred_tgt = self.Discriminator(tgt_mapping)
        loss_tgt = -pred_tgt.mean()
        return loss_tgt

