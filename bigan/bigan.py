# To use utils located in parent directory
import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path)

from utils import *


class BiGAN():
    def __init__(self, discriminator, generator, encoder, d_optim, g_e_optim,
                 z_dim, decay_lr_params=None, img_dim=28,
                 save_every=None, model_path=None, use_cuda=True):

        self.D = discriminator
        self.G = generator
        self.E = encoder
        self.d_optim = d_optim
        self.g_e_optim = g_e_optim
        if decay_lr_params is not None:
            self.decay = decay_lr_params['decay']
            self.decay_every = decay_lr_params['decay_every']
            self.init_lr = decay_lr_params['init_lr']
        self.img_dim = img_dim
        self.use_cuda = use_cuda
        self.save_every = save_every
        self.model_path = model_path
        self.model_nr = np.random.randint(1, 1e5)
        self.d_losses = []
        self.g_e_losses = []
        self.epoch = 0

        # self.D.apply(init_weight)
        # self.G.apply(init_weight)
        # self.E.apply(init_weight)

        if use_cuda:
            self.G.cuda()
            self.D.cuda()
            self.E.cuda()

    def train(self, train_loader, n_epochs):
        train_start = time.time()
        for epoch in range(1, n_epochs +1):
            self.set_to_train_mode()
            # Decay learning rate linearly to zero
            if self.decay and epoch % self.decay_every == 0:
                new_lr = self.init_lr * (1 - epoch /n_epochs)
                self._decay_lr(new_lr)

            print(f'Epoch: {epoch}/{n_epochs}')
            self._train_epoch(train_loader)

            # Save model every now and then
            if self.save_every is not None and epoch % self.save_every == 0:
                self.save_model(self.model_nr, epoch)
            elif self.save_every and epoch == n_epochs:
                self.save_model(self.model_nr, epoch)
            self.epoch += 1
        train_time = time.time() - train_start
        if n_epochs != 0:
            print(f'Training done in {int(train_time)}s.')

    def _train_epoch(self, train_loader):
        for batch in Bar(train_loader):
            batch = batch.cuda() if self.use_cuda else batch
            d_loss, g_e_loss = self._train_step(batch)
            self.d_losses.append(d_loss)
            self.g_e_losses.append(g_e_loss)

        epoch_d_losses = self.d_losses[-batch.size(0):]
        epoch_g_e_losses = self.g_e_losses[-batch.size(0):]
        mean_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
        mean_g_e_loss = sum(epoch_g_e_losses) / len(epoch_g_e_losses)
        print(f'd_loss: {round(mean_d_loss, 2)} | g_e_loss: {round(mean_g_e_loss, 2)}')

    def _train_step(self, batch):
        batch_size = batch.size(0)
        x_real = batch.view(batch_size, -1)             # x
        z_real = self.E(x_real)                         # E(x)
        real_comb = torch.cat((x_real, z_real), dim=1)  # (x, E(x))

        z_fake, x_fake = self.G(batch_size)
        fake_comb = torch.cat((x_fake, z_fake), dim=1)      # (G(z), z)

        # Discriminator predictions
        d_real = self.D(real_comb)       # D(x, E(x))
        d_fake = self.D(fake_comb)       # D(G(z), z)

        # Discriminator loss
        d_loss = -torch.mean(log(d_real) + log(1 - d_fake))

        # Generator/Encoder loss
        g_e_loss = -torch.mean(log(1 - d_real) + log(d_fake))

        d_loss.backward(retain_graph=True)
        self.d_optim.step()
        g_e_loss.backward()
        self.g_e_optim.step()

        self._reset_grad()

        return d_loss.item(), g_e_loss.item()

    def generate(self, n_samples):
        _, x = self.G(n_samples)
        return x.view(n_samples, self.img_dim, self.img_dim, 1) # (n, h, w, 1)

    def reconstruct(self, x):
        num_samples, h, w = x.size()
        x_flat = x.view(num_samples, -1)
        z = self.E(x_flat)
        _, recons_flat = self.G(num_samples, z)
        recons = recons_flat.view(num_samples, h, w, 1) # (n, h, w, 1)
        return recons

    def set_to_eval_mode(self):
        self.D.eval()
        self.G.eval()
        self.E.eval()

    def set_to_train_mode(self):
        self.D.train()
        self.G.train()
        self.E.train()

    def _reset_grad(self):
        self.d_optim.zero_grad()
        self.g_e_optim.zero_grad()

    def _decay_lr(self, new_lr):
        for optimizer in [self.d_optim, self.g_e_optim]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

    def save_model(self, model_nr, epoch):
        path = f'{self.model_path}bigan_{model_nr}_epoch{epoch}'
        torch.save({
            'model_nr': model_nr,
            'epoch': epoch,
            'd_state_dict': self.D.state_dict(),
            'g_state_dict': self.G.state_dict(),
            'e_state_dict': self.E.state_dict(),
            'd_optim_state_dict': self.d_optim.state_dict(),
            'g_e_optim_state_dict': self.g_e_optim.state_dict(),
            'd_losses': self.d_losses,
            'g_e_losses': self.g_e_losses,
        }, path)
        print("- " *50)
        print(f"Model saved as 'bigan_{model_nr}_epoch{epoch}'.")
        print("- " *50)

    def load_model(self, path):
        map_location = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        checkpoint = torch.load(path, map_location=map_location)
        self.model_nr = checkpoint['model_nr']
        self.epoch = checkpoint['epoch']
        self.D.load_state_dict(checkpoint['d_state_dict'])
        self.G.load_state_dict(checkpoint['g_state_dict'])
        self.E.load_state_dict(checkpoint['e_state_dict'])
        self.d_optim.load_state_dict(checkpoint['d_optim_state_dict'])
        self.g_e_optim.load_state_dict(checkpoint['g_e_optim_state_dict'])
        self.d_losses = checkpoint['d_losses']
        self.g_e_losses = checkpoint['g_e_losses']
        self.G.train(), self.D.train(), self.E.train()
        print \
            (f"Model loaded. [Trained for {self.epoch} epochs | Current d_loss: {round(self.d_losses[-1], 2)} | Current g_e_loss: {round(self.g_e_losses[-1], 2)}]")
        print("- " *80)
