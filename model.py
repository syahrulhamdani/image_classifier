import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.optim as optim
from collections import OrderedDict
import argparse


class Classifier(nn.Module):
    """Define new classifier."""
    def __init__(self, input_size, hidden_sizes, output_size, drop_p=0.5):
        """Initialize a new classifier to be attached to pretrained models.

        parameters
        ----------
        hidden_sizes: list of int. Number of int shows number of hidden layers,
        the ints show the layer sizes.
        output_size: int. Sizes of the output layer
        """
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(
            input_size, hidden_sizes[0]
        )])
        layers = zip(hidden_sizes[:-1], hidden_sizes[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layers])
        self.output = nn.Linear(hidden_sizes[-1], output_size)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        """Forward pass through the network. Returns logits."""
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)


def load_pretrained(arch, hidden_sizes, output_size, drop_p=0.5):
    """load pretrained models."""
    model = models.__dict__[arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    input_size = model.classifier[0].in_features
    new_classifier = Classifier(input_size, hidden_sizes, output_size, drop_p)
    model.classifier = new_classifier

    return model
