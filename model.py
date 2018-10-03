import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.optim as optim
from collections import OrderedDict
import argparse


class Model():
    """Model wrapper"""
    def __init__(self):
        pass

# TODO:
# 1. function load_pretrained with classifier and pretrained model name
# 2. function deep_learning to train the model
# 3. function validate_model to validate the model used in deep_learning
#    function
# 4. function test_model to test the model after being trained
