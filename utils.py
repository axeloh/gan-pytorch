"""Collection of common imports and methods """

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from scipy.stats import norm
from tqdm import trange, tqdm_notebook
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import StepLR
import os
import os.path as osp
import time
from barbar import Bar
from datetime import date
from sklearn.metrics import accuracy_score
from tqdm import trange, tqdm


def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

def scale(x, a, b):
    """Scale data between a and b, a<b"""
    x = (b-a)*((x-x.min())/(x.max()-x.min())) + a
    return x

def log(x):
    return torch.log(x + 1e-8)
