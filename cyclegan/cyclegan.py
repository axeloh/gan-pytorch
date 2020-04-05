from utils import *

class CycleGAN(nn.Module):
    """CycleGAN model, inspired by https://github.com/yunjey/mnist-svhn-transfer/blob/master/solver.py"""

    def __init__(self, g_ab, g_ba, d_a, d_b, g_optim, d_optim, config):
        super().__init__()
        self.g_ab = g_ab  # mnist  -> cmnist      (generator)
        self.g_ba = g_ba  # cmnist -> mnist       (generator)
        self.d_a = d_a  # classifying mnist     (discriminator)
        self.d_b = d_b  # classifying cmnist    (discriminator)
        self.g_optim = g_optim  # generators optimizer
        self.d_optim = d_optim  # discriminators optimizer

        self.use_recon_loss = config['use_recon_loss']
        self.lambda1 = config['lambda1']
        self.lambda2 = config['lambda2']
        self.model_path = config['model_path']
        self.use_cuda = config['use_cuda']
        self.use_labels = config['use_labels']
        self.num_classes = config['num_classes']

        self.device = torch.device('cuda') if self.use_cuda else None
        self.set_device(self.device)
        self.d_losses = []
        self.g_losses = []

    def train(self, mnist_loader, cmnist_loader, n_epochs):
        for epoch in range(1, n_epochs + 1):
            self._train_epoch(mnist_loader, cmnist_loader, epoch)

    def _train_epoch(self, mnist_loader, cmnist_loader, epoch):
        epoch_start = time.time()
        mnist_iter = iter(mnist_loader)
        cmnist_iter = iter(cmnist_loader)
        iter_per_epoch = min(len(mnist_iter), len(cmnist_iter))  # In case different size of datasets
        for _ in range(iter_per_epoch):
            d_loss, g_loss = self._train_step(mnist_iter, cmnist_iter)
            self.d_losses.append(d_loss)
            self.g_losses.append(g_loss)

        log_string = f'Epoch: {epoch} [{int(time.time() - epoch_start)}s]| '
        log_string += f'Discriminator loss: {d_loss:.3f}, Generator loss: {g_loss:.3f}'
        print(log_string)

    def _train_step(self, mnist_iter, svhn_iter):
        m_batch, m_labels = mnist_iter.next()
        s_batch, s_labels = svhn_iter.next()
        min_ = min(len(m_batch), len(s_batch))  # In case different size of last batch
        a_real = m_batch.to(self.device)[:min_]
        b_real = s_batch.to(self.device)[:min_]
        if self.use_labels:
            a_labels = m_labels.long()[:min_].to(self.device)
            b_labels = s_labels.long()[:min_].to(self.device)
            # For mnist: real labels range from 0 to 9, fake label will be 10
            fake_labels = torch.tensor([self.num_classes] * min_).long().to(self.device)
            criterion = nn.CrossEntropyLoss()

        # ------------- Update g_ab and g_ba -------------#
        # Discriminators requires no gradient when optimizing generators
        self.set_requires_grad([self.d_a, self.d_b], False)
        self.reset_grad()

        # Compute fake images and reconstruction images.
        b_fake = self.g_ab(a_real)
        a_recon = self.g_ba(b_fake)
        a_fake = self.g_ba(b_real)
        b_recon = self.g_ab(a_fake)

        d_b_fake = self.d_b(b_fake)
        d_a_fake = self.d_a(a_fake)

        if self.use_labels:
            g_ab_loss = criterion(d_b_fake, a_labels)
            g_ba_loss = criterion(d_a_fake, b_labels)
        else:
            g_ab_loss = torch.mean((d_b_fake - 1) ** 2)
            g_ba_loss = torch.mean((d_a_fake - 1) ** 2)

        g_loss = g_ab_loss + g_ba_loss
        if self.use_recon_loss:
            aba_cycle_loss = torch.mean((a_recon - a_real).abs()) * self.lambda1
            bab_cycle_loss = torch.mean((b_recon - b_real).abs()) * self.lambda2
            g_loss += aba_cycle_loss + bab_cycle_loss

        g_loss.backward()
        self.g_optim.step()

        # ------------- Update d_a and d_b -------------#
        self.set_requires_grad([self.d_a, self.d_b], True)
        self.reset_grad()

        # Compute fake images and reconstruction images.
        b_fake = self.g_ab(a_real)
        a_fake = self.g_ba(b_real)

        # d_a
        d_a_real = self.d_a(a_real)
        d_a_fake = self.d_a(a_fake)
        if self.use_labels:
            d_a_loss = criterion(d_a_real, a_labels)
            d_a_loss += criterion(d_a_fake, fake_labels)
        else:
            d_a_loss = torch.mean((d_a_real - 1) ** 2 + d_a_fake ** 2)

        # d_b
        d_b_real = self.d_b(b_real)
        d_b_fake = self.d_b(b_fake)
        if self.use_labels:
            d_b_loss = criterion(d_b_real, b_labels)
            d_b_loss += criterion(d_b_fake, fake_labels)
        else:
            d_b_loss = torch.mean((d_b_real - 1) ** 2 + d_b_fake ** 2)

        d_loss = d_a_loss + d_b_loss
        d_loss.backward()
        self.d_optim.step()

        self.d_losses.append(d_loss.item())
        self.g_losses.append(g_loss.item())

        return d_loss.item(), g_loss.item()

    def reset_grad(self):
        self.g_optim.zero_grad()
        self.d_optim.zero_grad()

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