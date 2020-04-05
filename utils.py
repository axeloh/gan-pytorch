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
import scipy.ndimage
from PIL import Image as PILImage
import cv2


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

def plot_cyclegan_samples_and_recons(m1, c1, m2, c2, m3, c3, dname1, dname2):
    m1, m2, m3 = m1.repeat(3, axis=3), m2.repeat(3, axis=3), m3.repeat(3, axis=3)
    a_reconstructions = np.concatenate([m1, c1, m2], axis=0)
    b_reconstructions = np.concatenate([c2, m3, c3], axis=0)

    show_samples(a_reconstructions * 255.0, nrow=20,
                 fname=f'results/{dname1}.png',
                 title=f'Source domain: {dname1.upper()}')
    show_samples(b_reconstructions * 255.0, nrow=20,
                 fname=f'results/{dname2}.png',
                 title=f'Source domain: {dname2.upper()}')


def get_colored_mnist(data):
    # from https://www.wouterbulten.nl/blog/tech/getting-started-with-gans-2-colorful-mnist/
    # Read Lena image
    lena = PILImage.open('../cyclegan/results/lena.jpg')

    # Resize
    batch_resized = np.asarray([scipy.ndimage.zoom(image, (2.3, 2.3, 1), order=1) for image in data])

    # Extend to RGB
    batch_rgb = np.concatenate([batch_resized, batch_resized, batch_resized], axis=3)

    # Make binary
    batch_binary = (batch_rgb > 0.5)

    batch = np.zeros((data.shape[0], 28, 28, 3))

    for i in range(data.shape[0]):
        # Take a random crop of the Lena image (background)
        x_c = np.random.randint(0, lena.size[0] - 64)
        y_c = np.random.randint(0, lena.size[1] - 64)
        image = lena.crop((x_c, y_c, x_c + 64, y_c + 64))
        image = np.asarray(image) / 255.0

        # Invert the colors at the location of the number
        image[batch_binary[i]] = 1 - image[batch_binary[i]]

        batch[i] = cv2.resize(image, (0, 0), fx=28 / 64, fy=28 / 64, interpolation=cv2.INTER_AREA)
    return batch.transpose(0, 3, 1, 2)