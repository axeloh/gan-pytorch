import sys
from utils import *

class GAN1d(nn.Module):
    """ Simple GAN for 1d toy data"""
    def __init__(self, data_dim, noise_dim, hid_dim=8, use_cuda=True):
        super().__init__()
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.use_cuda = use_cuda

        self.generator = nn.Sequential(
            nn.Linear(noise_dim, hid_dim), nn.LeakyReLU(),
            nn.Linear(hid_dim, hid_dim), nn.LeakyReLU(),
            nn.Linear(hid_dim, hid_dim), nn.LeakyReLU(),
            nn.Linear(hid_dim, data_dim)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(data_dim, hid_dim*2), nn.LeakyReLU(),
            nn.Linear(hid_dim*2, hid_dim*2), nn.LeakyReLU(),
            nn.Linear(hid_dim*2, 1), nn.Sigmoid()
        )

        self.generator.apply(init_weight)
        self.discriminator.apply(init_weight)

    def _sample_noise(self, num_samples):
        """
        Sample noise/latent to generate examples from
        Sampling from U(-1,1)
        """
        noise = torch.zeros((num_samples, self.noise_dim)).float().uniform_(-1, 1)
        noise = noise.cuda() if self.use_cuda else noise
        return noise

    def G(self, num_samples):
        """ Transforms random input to data instances through generator network"""
        noise = self._sample_noise(num_samples)
        return self.generator(noise)

    def D(self, x):
        """
        Takes in real or fake instance
        Returns a probability (0 for fake, 1 for real)
        """
        return self.discriminator(x)
