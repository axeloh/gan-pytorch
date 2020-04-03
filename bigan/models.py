import torch
import torch.nn as nn

class Generator(nn.Module):
    """ Generator.
    Maps latent codes z -> flattened image x
    z_dim: dimension of latent variables, typically 50
    out_dim: h*w (flattened image)
    """
    def __init__(self, z_dim, hid_dim, img_dim, use_cuda=True):
        super().__init__()
        self.z_dim = z_dim
        out_dim = img_dim * img_dim
        self.device = torch.device("cuda") if use_cuda else None

        self.fc = nn.Sequential(
            nn.Linear(z_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim, affine=False),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
            nn.Tanh()
        )

    def _sample_z(self, num_samples):
        """ Sample z from N(0,1)"""
        return torch.randn((num_samples, self.z_dim), device=self.device)

    def forward(self, num_samples, z=None):
        if z is None:
            z = self._sample_z(num_samples)
        x = self.fc(z)
        return z, x

class Encoder(nn.Module):
    """ Encoder.
    Maps from flattened image x to latent code z
    in_dim: h*w (flattened image)
    z_dim: dimension of latent variables, typically 50
    """
    def __init__(self, img_dim, hid_dim, z_dim):
        super().__init__()
        in_dim = img_dim * img_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim, affine=False),
            nn.LeakyReLU(0.2),
            nn.Linear(hid_dim, z_dim)
        )

    def forward(self, x):
        z = self.fc(x)
        return z

class Discriminator(nn.Module):
    """ Discriminator.
    Takes in flattened image (real or fake), along with latents (real or fake)
    in_dim: h*w + z_dim
    Outputs y [0,1] (0 fake, 1 real)
    """
    def __init__(self, img_dim, z_dim, hid_dim):
        super().__init__()
        in_dim = img_dim * img_dim + z_dim

        self.fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hid_dim, hid_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hid_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return y
