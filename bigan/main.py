import sys
import os
import warnings
import argparse
# To use utils located in parent directory
warnings.filterwarnings("ignore")
path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path)

from utils import *
from torchvision.datasets import MNIST, FashionMNIST
from bigan_models import Generator, Encoder, Discriminator, LinearClassifier
from bigan import BiGAN


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion'])
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--z_dim',  type=int, default=50)
parser.add_argument('--hid_dim',  type=int, default=1024)
parser.add_argument('--lr',  type=int, default=2e-4)
parser.add_argument('--use_cuda',  type=str, default='True')
parser.add_argument('--save_every',  type=int, default=30)
parser.add_argument('--pretrained_name',  type=str, default='None')


args = parser.parse_args()

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

if args.dataset == 'mnist':
    train_data = MNIST(root='data/mnist', train=True, download=True, transform=transform)
    test_data = MNIST(root='data/mnist', train=False, download=True, transform=transform)

if args.dataset == 'fashion':
    train_data = FashionMNIST(root='data/fashion', train=True, download=True, transform=transform)
    test_data = FashionMNIST(root='data/fashion', train=False, download=True, transform=transform)

# BiGAN params
z_dim = args.z_dim
hid_dim = args.hid_dim

# Train params
use_cuda = args.use_cuda and torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device('cpu')
n_epochs = args.n_epochs
batch_size = args.batch_size
lr = args.lr
beta1, beta2 = 0.5, 0.999
weight_decay = 2.5e-5
decay_lr_params = {'decay': True,
                   'decay_every': 1,
                   'init_lr': lr}

model_path = './saved_models/'
save_every = args.save_every
use_pretrained = args.pretrained_name != 'None'

# Load and scale data
train_x = train_data.data.float()
test_x = test_data.data.float()
train_x = scale(train_x, -1, 1)
test_x = scale(test_x, -1, 1)
test_y = test_data.targets.long()

train_x_loader = torch.utils.data.DataLoader(train_x, batch_size=batch_size, shuffle=True)
img_dim = train_x.size(1)

# Models
D = Discriminator(img_dim, z_dim, hid_dim)
G = Generator(z_dim, hid_dim, img_dim, use_cuda=use_cuda)
E = Encoder(img_dim, hid_dim, z_dim)

# Optimizers
d_optim = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
g_e_optim = torch.optim.Adam(list(G.parameters()) + list(E.parameters()), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

# BiGAN
bigan = BiGAN(D, G, E, d_optim, g_e_optim, z_dim,
              decay_lr_params, model_path=model_path,
              save_every=save_every, use_cuda=use_cuda)

if use_pretrained:
    bigan.load_model(model_path + args.pretrained_name)

#### TRAIN ####
if not use_pretrained:
    print(f"Training BiGAN model for {n_epochs} epochs..")
    bigan.train(train_x_loader, n_epochs)


#### EVALUATE ####
bigan.set_to_eval_mode()
with torch.no_grad():
    # Samples
    samples = bigan.generate(100)
    samples = scale(samples, 0, 1)

    # Real-reconstructed pairs of 20 images from test set
    real_imgs = test_x[:20]
    real_imgs = real_imgs.cuda() if use_cuda else real_imgs
    recons = bigan.reconstruct(real_imgs)
    real_imgs = scale(real_imgs, 0, 1)
    recons = scale(recons, 0, 1)
    real_recon_pairs = torch.cat((real_imgs.unsqueeze(-1), recons), dim=0)

gan_losses, samples, reconstructions = np.array(bigan.d_losses), np.array(samples.cpu()), np.array(real_recon_pairs.cpu())


# Show real samples
imgs = train_x[:100]
show_samples(imgs.reshape([100, 28, 28, 1]) * 255.0, title=f'MNIST samples')

# Plot GAN loss during training
plot_gan_training(gan_losses, 'Losses', './results/gan_losses.png')

# Show samples and reconstruction-pairs
show_samples(samples * 255.0, fname=f'results/samples_epoch{bigan.epoch}.png', title=f'BiGAN generated samples after {bigan.epoch} epochs')
show_samples(reconstructions * 255.0, nrow=20, fname=f'results/reconstructions_epochs{bigan.epoch}.png', title=f'BiGAN reconstructions after {bigan.epoch} epochs')



#### Train and evaluate linear classifier for classifying digits ####
#### Compare trained encoder to random encoder ####
#### One uses the trained encoder from BiGAN, the other uses a random encoder ####
clf_n_epochs = 10
data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Using Random encoder
print("Training classifier with random encoder..")
E_random = Encoder(img_dim, hid_dim, z_dim).to(device)
clf_random = LinearClassifier(E_random, z_dim, use_cuda=use_cuda)
clf_random.train(data_loader, test_loader, clf_n_epochs)

# Using Trained encoder
print("Training classifier with trained BiGAN encoder..")
E_pretrained = bigan.E
clf_pretrained = LinearClassifier(E_pretrained, z_dim, use_cuda=use_cuda)
clf_pretrained.train(data_loader, test_loader, clf_n_epochs)

# Cross Entropy Losses on test data during training
pretrained_losses = np.array(clf_pretrained.val_losses)
random_losses = np.array(clf_random.val_losses)
plot_bigan_supervised(pretrained_losses, random_losses, title='Linear classification losses', fname='results/supervised_losses.png')


# Evaluate accuracy on test data
random_preds = clf_random.predict(test_x.to(device)).cpu()
random_accuracy = accuracy_score(test_y, random_preds)
print(f"Final test accuracy using Random encoder: {random_accuracy}")

pretrained_preds = clf_pretrained.predict(test_x.to(device)).cpu()
pretrained_accuracy = accuracy_score(test_y, pretrained_preds)
print(f"Final test accuracy using BiGAN encoder: {pretrained_accuracy}")

