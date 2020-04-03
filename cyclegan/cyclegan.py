import torch
import torch.nn as nn
import time
from barbar import Bar
from utils import *

class CycleGAN(nn.Module):
    """CycleGAN model, inspired by https://github.com/yunjey/mnist-svhn-transfer/blob/master/solver.py"""
    def __init__(self, g_ab, g_ba, d_a, d_b, g_optim, d_optim, config):
        super().__init__()
        self.g_ab = g_ab            # mnist  -> cmnist      (generator)
        self.g_ba = g_ba            # cmnist -> mnist       (generator)
        self.d_a = d_a              # classifying mnist     (discriminator)
        self.d_b = d_b              # classifying cmnist    (discriminator)
        self.g_optim = g_optim      # generators optimizer
        self.d_optim = d_optim      # discriminators optimizer

        self.use_recon_loss = config['use_recon_loss']
        self.lambda_ = config['lambda'] # importance of cycle loss
        self.use_cuda = config['use_cuda']
        self.device = torch.device('cuda') if self.use_cuda else None
        self.set_device(self.device)
        self.curr_epoch = 0
        self.log_interval = 100
        self.d_losses = []
        self.g_losses = []

    def train(self, mnist_loader, cmnist_loader, n_epochs):
        for epoch in range(1, n_epochs +1):
            self._train_epoch(mnist_loader, cmnist_loader, epoch)

    def _train_epoch(self, mnist_loader, cmnist_loader, epoch):
        epoch_start = time.time()
        mnist_iter = iter(mnist_loader)
        cmnist_iter = iter(cmnist_loader)
        iter_per_epoch = min(len(mnist_iter), len(cmnist_iter)) # In case different size of datasets
        for _ in tqdm(range(iter_per_epoch)):
            m_batch, _ = mnist_iter.next()
            cm_batch, _ = cmnist_iter.next()
            d_loss, g_loss = self._train_step(m_batch, cm_batch)

        log_string = f'Epoch: {epoch} [{int(time.time( ) -epoch_start)}s]| '
        log_string += f'Discriminator loss: {d_loss:.3f}, Generator loss: {g_loss:.3f}'
        print(log_string)

    def _train_step(self, m_real, cm_real):
        a_real = m_real.to(self.device)
        b_real = cm_real.to(self.device)

        # Compute fake images and reconstruction images.
        b_fake = self.g_ab(a_real)
        a_recon = self.g_ba(b_fake)
        a_fake = self.g_ba(b_real)
        b_recon = self.g_ab(a_fake)

        d_b_fake = self.d_b(b_fake)
        d_b_real = self.d_b(b_real)
        d_a_fake = self.d_a(a_fake)
        d_a_real = self.d_a(a_real)

        # ----------- Update discriminators -----------#
        self.d_optim.zero_grad()
        d_b_loss = torch.mean((d_b_real - 1 )**2 + d_b_fake**2)
        d_a_loss = torch.mean((d_a_real - 1 )**2 + d_a_fake**2)

        d_loss = d_b_loss + d_a_loss
        d_loss.backward(retain_graph=True)
        self.d_optim.step()

        # ----------- Update generators -----------#
        self.g_optim.zero_grad()
        g_ab_loss = torch.mean((d_b_fake - 1 )**2)
        g_ba_loss = torch.mean((d_a_fake - 1 )**2)
        g_loss = g_ab_loss + g_ba_loss
        if self.use_recon_loss:
            a_recon_loss = torch.mean((a_real - a_recon).abs())
            b_recon_loss = torch.mean((b_real - b_recon).abs())
            g_loss += self.lambda_ * (a_recon_loss + b_recon_loss)

        g_loss.backward()
        self.g_optim.step()

        self.d_losses.append(d_loss.item())
        self.g_losses.append(g_loss.item())

        return d_loss.item(), g_loss.item()

    def set_to_train_mode(self):
        self.g_ab.train()
        self.g_ba.train()
        self.d_a.train()
        self.d_b.train()

    def set_to_eval_mode(self):
        self.g_ab.eval()
        self.g_ba.eval()
        self.d_a.eval()
        self.d_b.eval()

    def set_device(self, device):
        self.g_ab = self.g_ab.to(device)
        self.g_ba = self.g_ba.to(device)
        self.d_a = self.d_a.to(device)
        self.d_b = self.d_b.to(device)


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
