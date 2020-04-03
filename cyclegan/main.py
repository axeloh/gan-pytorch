
from utils import *
from torchvision.datasets import MNIST, SVHN
from cyclegan_models import G_AB, G_BA, D_A, D_B
from cyclegan import CycleGAN


transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

# Load and scale data
mnist_data = MNIST(root='data/mnist', train=True, download=True, transform=transform)  # (n, 1, 28, 28)
cmnist_data = SVHN(root='data/svhn', split='train', download=True, transform=transform)  # (n, 3, 28, 28)

print(mnist_data)
print(cmnist_data)

#cmnist_data = torch.from_numpy(cmnist_data)  # Because SVHN comes as numpy array
#mnist_data = mnist_data.unsqueeze(dim=1)  # To get (n, 1, 28, 28)
#mnist_data = scale(mnist_data, -1, 1).float()
#cmnist_data = scale(cmnist_data, -1, 1).float()

#print(mnist_data.shape)
#print(cmnist_data.shape)

# CycleGAN params
n_filters = 32
use_recon_loss = True
lambda_ = 10

# Train-params
use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device('cpu')
n_epochs = 1
batch_size = 128
lr = 2e-4
beta1, beta2 = 0.9, 0.999
weight_decay = 0  # 2.5e-5

config = {
    'use_recon_loss': use_recon_loss,
    'lambda': lambda_,
    'use_cuda': use_cuda
}

# Data loaders
mnist_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)
cmnist_loader = torch.utils.data.DataLoader(cmnist_data, batch_size=batch_size, shuffle=True)

# Models
g_ab = G_AB(n_filters=n_filters)
g_ba = G_BA(n_filters=n_filters)
d_a = D_A(n_filters=n_filters)
d_b = D_B(n_filters=n_filters)

# Optimizers
g_optim = torch.optim.Adam(list(g_ab.parameters()) + list(g_ba.parameters()), lr=lr, betas=(beta1, beta2),
                           weight_decay=weight_decay)
d_optim = torch.optim.Adam(list(d_a.parameters()) + list(d_b.parameters()), lr=lr, betas=(beta1, beta2),
                           weight_decay=weight_decay)

# CycleGAN
cyclegan = CycleGAN(g_ab, g_ba, d_a, d_b, g_optim, d_optim, config)


# --------- Train ---------#
print(f"Training CycleGAN model for {n_epochs} epochs..\n")
cyclegan.train(mnist_loader, cmnist_loader, n_epochs)


# --------- Evaluate ---------#
cyclegan.set_to_eval_mode()
with torch.no_grad():
    mnist_real = mnist_data[:20].to(device)
    cmnist_fake = g_ab(mnist_real)
    mnist_recon = g_ba(cmnist_fake)

    cmnist_real = cmnist_data[:20].to(device)
    mnist_fake = g_ba(cmnist_real)
    cmnist_recon = g_ab(mnist_fake)

    # Scale and permute to (n, h, w, 1)
    mnist_real = scale(mnist_real, 0, 1).permute(0, 2, 3, 1)
    cmnist_fake = scale(cmnist_fake, 0, 1).permute(0, 2, 3, 1)
    mnist_recon = scale(mnist_recon, 0, 1).permute(0, 2, 3, 1)
    cmnist_real = scale(cmnist_real, 0, 1).permute(0, 2, 3, 1)
    mnist_fake = scale(mnist_fake, 0, 1).permute(0, 2, 3, 1)
    cmnist_recon = scale(cmnist_recon, 0, 1).permute(0, 2, 3, 1)

d_losses, g_losses = cyclegan.d_losses, cyclegan.g_losses
plt.title("Discriminator and Generator loss")
plt.xlabel('Training iteration')
plt.ylabel('Loss')
plt.plot([i for i in range(len(d_losses))], d_losses, label='d_loss')
plt.plot([i for i in range(len(g_losses))], g_losses, label='g_loss')
plt.legend()
plt.show()

#return np.array(mnist_real.cpu()), np.array(cmnist_fake.cpu()), np.array(mnist_recon.cpu()), np.array(
 #   cmnist_real.cpu()), np.array(mnist_fake.cpu()), np.array(cmnist_recon.cpu())