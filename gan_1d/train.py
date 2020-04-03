
from load_datasets import *
from utils import *
from gan1d import GAN1d


def train(train_data, use_non_saturing):
    """
    Inputs
    - train_data: An (20000, 1) numpy array of floats in [-1, 1]
    - use_non_saturing: Whether to use the non-saturating GAN objective or the original minimax GAN objective

    Returns
    - a (# of training iterations,) numpy array of discriminator losses evaluated every minibatch
    - a numpy array of size (5000,) of results drawn from your model at epoch #1
    - a numpy array of size (1000,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (1000,), corresponding to the discriminator output (after sigmoid)
        at each location in the previous array at epoch #1

    - a numpy array of size (5000,) of results drawn from your model at the end of training
    - a numpy array of size (1000,) linearly spaced from [-1, 1]; hint: np.linspace
    - a numpy array of size (1000,), corresponding to the discriminator output (after sigmoid)
        at each location in the previous array at the end of training
    """

    print(train_data.shape)
    train_data = torch.from_numpy(train_data).float()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda') if use_cuda else None
    gan = GAN1d(data_dim=1, noise_dim=5, hid_dim=16, use_cuda=use_cuda).to(device)

    n_epochs = 40
    k = 4
    d_optimizer = torch.optim.Adam(gan.discriminator.parameters(), lr=1e-4, betas=(0, 0.9))
    g_optimizer = torch.optim.Adam(gan.generator.parameters(), lr=1e-4, betas=(0, 0.9))
    d_losses = []
    g_losses = []

    # Training using the original MinMax GAN Objective
    train_start = time.time()
    for epoch in range(n_epochs):
        for batch in train_loader:
            num_samples = batch.shape[0]
            batch = batch.cuda() if use_cuda else batch
            # k steps of training the discriminator/critic
            # Maximize log D(x) + log(1 - D(G(z)))
            for _ in range(k):
                fake_samples = gan.G(num_samples)
                d_fake = gan.D(fake_samples)
                d_real = gan.D(batch)
                d_loss = -torch.mean(d_real.log() + (1 - d_fake).log())

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
            d_losses.append(d_loss.item())

            # Training generator
            # Minimize log(1 - D(G(z)))
            fake_samples = gan.G(num_samples)
            d_fake = gan.D(fake_samples)
            if use_non_saturing:
                g_loss = -torch.mean(d_fake.log())
            else:
                g_loss = torch.mean((1 - d_fake).log())

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            g_losses.append(g_loss)

        if epoch % 5 == 0:
            print(f'{epoch + 1}/{n_epochs} epochs')

        if epoch == 0:
            with torch.no_grad():
                samples1 = gan.G(5000)
                xs1 = torch.from_numpy(np.linspace(start=-1, stop=1, num=1000)).unsqueeze(1).float()
                xs1 = xs1.cuda() if use_cuda else xs1
                ys1 = gan.D(xs1)
    print(f'Training done in {int(time.time() - train_start)}s.')

    # Sample
    with torch.no_grad():
        samples_end = gan.G(5000)
        xs_end = torch.from_numpy(np.linspace(start=-1, stop=1, num=1000)).unsqueeze(1).float()
        xs_end = xs_end.cuda() if use_cuda else xs_end
        ys_end = gan.D(xs_end)

    return np.array(d_losses), np.array(samples1.cpu()), np.array(xs1.cpu()), np.array(ys1.cpu()), np.array(samples_end.cpu()), np.array(xs_end.cpu()), np.array(ys_end.cpu())


if __name__ == '__main__':
    # Load toy data
    train_data = load_data1()
    train_data2 = load_data2()
    warmup_data = load_warmup_data()

    # Train
    train(train_data, True)




