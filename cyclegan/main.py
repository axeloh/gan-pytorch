import argparse
import warnings
import os
import sys
warnings.filterwarnings("ignore")
# To use utils located in parent directory
path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path)

from utils import *
from torchvision.datasets import MNIST, SVHN
from cyclegan_models import G_AB, G_BA, D_A, D_B
from cyclegan import CycleGAN



def str_to_bool(s):
    return s.lower() in ['true']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--g_n_filters', type=int, default=64)
    parser.add_argument('--d_n_filters', type=int, default=64)
    parser.add_argument('--lambda1', type=int, default=1)
    parser.add_argument('--lambda2', type=int, default=1)
    parser.add_argument('--use_recon_loss', type=str_to_bool, default=True)
    parser.add_argument('--use_labels', type=str_to_bool, default=False)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--pretrained_name', type=str, default=None)
    parser.add_argument('--use_colored_mnist', type=str_to_bool, default=False)

    # Train params
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=int, default=2e-4)
    parser.add_argument('--beta1', type=int, default=0.5)
    parser.add_argument('--beta2', type=int, default=0.999)
    parser.add_argument('--weight_decay', type=int, default=2.5e-4)
    parser.add_argument('--use_cuda', type=str_to_bool, default=True)
    parser.add_argument('--model_path', type=str, default="./saved_models/")

    args = parser.parse_args()
    print(args)

    # Model params
    g_n_filters = args.g_n_filters
    d_n_filters = args.d_n_filters
    use_cuda = args.use_cuda and torch.cuda.is_available()
    config = {
        'use_recon_loss': args.use_recon_loss,
        'lambda1': args.lambda1,
        'lambda2': args.lambda2,
        'model_path': args.model_path,
        'use_cuda': use_cuda,
        'use_labels': args.use_labels,
        'num_classes': args.num_classes
    }

    # Train params
    device = torch.device("cuda") if use_cuda else torch.device('cpu')
    n_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    beta1, beta2 = args.beta1, args.beta2
    weight_decay = args.weight_decay

    # Load data
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_data = MNIST(root='data/mnist', train=True, download=True, transform=transform)
    if args.use_colored_mnist:
        # TODO
        # Variable name still svhn, but now using colored mnist instead
        svhn = get_colored_mnist(np.array(mnist_data.data.reshape(-1, 28, 28, 1) / 255.0))
        labels = mnist_data.targets
        svhn_data = torch.from_numpy(svhn), labels
    else:
        svhn_data = SVHN(root='data/svhn', split='train', download=True, transform=transform)

    mnist_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)
    svhn_loader = torch.utils.data.DataLoader(svhn_data, batch_size=batch_size, shuffle=True)

    # Models
    g_ab = G_AB(g_n_filters)
    g_ba = G_BA(g_n_filters)
    d_a = D_A(d_n_filters, args.use_labels)
    d_b = D_B(d_n_filters, args.use_labels)

    g_optim = torch.optim.Adam(list(g_ab.parameters()) + list(g_ba.parameters()),
                               lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    d_optim = torch.optim.Adam(list(d_a.parameters()) + list(d_b.parameters()),
                               lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

    # CycleGAN
    cyclegan = CycleGAN(g_ab, g_ba, d_a, d_b, g_optim, d_optim, config)


    # --------- Train ---------#
    print(f"Training CycleGAN model for {n_epochs} epochs..\n")
    cyclegan.train(mnist_loader, svhn_loader, n_epochs)

    # --------- Evaluate ---------#
    cyclegan.set_to_eval_mode()
    mnist_test = MNIST(root='data/mnist', train=False, download=True, transform=transform)  # (n, 1, 32, 32)
    svhn_test = SVHN(root='data/svhn', split='test', download=True, transform=transform)  # (n, 3, 32, 32)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_test, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        mnist_real, _ = iter(mnist_test_loader).next()
        mnist_real = mnist_real[10:30]
        mnist_real = scale(mnist_real, -1, 1).to(device)
        svhn_fake = g_ab(mnist_real)
        mnist_recon = g_ba(svhn_fake)

        svhn_real, _ = iter(svhn_test_loader).next()
        svhn_real = svhn_real[10:30]
        svhn_real = scale(svhn_real, -1, 1).to(device)
        mnist_fake = g_ba(svhn_real)
        svhn_recon = g_ab(mnist_fake)

        # Scale and permute to (n, h, w, c)
        mnist_real = scale(mnist_real, 0, 1).permute(0, 2, 3, 1)
        svhn_fake = scale(svhn_fake, 0, 1).permute(0, 2, 3, 1)
        mnist_recon = scale(mnist_recon, 0, 1).permute(0, 2, 3, 1)
        svhn_real = scale(svhn_real, 0, 1).permute(0, 2, 3, 1)
        mnist_fake = scale(mnist_fake, 0, 1).permute(0, 2, 3, 1)
        svhn_recon = scale(svhn_recon, 0, 1).permute(0, 2, 3, 1)

    d_losses, g_losses = cyclegan.d_losses, cyclegan.g_losses
    plt.title("Discriminator and Generator loss")
    plt.xlabel('Training iteration')
    plt.ylabel('Loss')
    plt.plot([i for i in range(len(d_losses))], d_losses, label='d_loss')
    plt.plot([i for i in range(len(g_losses))], g_losses, label='g_loss')
    plt.legend()
    plt.show()

    m1, c1, m2, c2, m3, c3 = np.array(mnist_real.cpu()), np.array(svhn_fake.cpu()), \
                             np.array(mnist_recon.cpu()), np.array(svhn_real.cpu()), \
                             np.array(mnist_fake.cpu()), np.array(svhn_recon.cpu())
    plot_cyclegan_samples_and_recons(m1, c1, m2, c2, m3, c3, 'mnist', 'svhn')