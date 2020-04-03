"""Collection of common imports and methods """

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
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
from torchvision.utils import make_grid
from os.path import join, dirname, exists
import torchvision.transforms as transforms


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

def savefig(fname, show_figure=False):
    if not exists(dirname(fname)):
        os.makedirs(dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()

def show_samples(samples, fname=None, nrow=10, title='Samples'):
    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')

    if fname is not None:
        savefig(fname)
    plt.show()

def plot_gan_training(losses, title, fname):
    plt.figure()
    n_itr = len(losses)
    xs = np.arange(n_itr)

    plt.plot(xs, losses, label='loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')
    savefig(fname)


def plot_bigan_supervised(pretrained_losses, random_losses, title, fname):
    plt.figure()
    xs = np.arange(len(pretrained_losses))
    plt.plot(xs, pretrained_losses, label='bigan')
    xs = np.arange(len(random_losses))
    plt.plot(xs, random_losses, label='random init')
    plt.legend()
    plt.title(title)
    savefig(fname)
    plt.show()

