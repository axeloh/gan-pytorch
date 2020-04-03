
from utils import *
from models import Generator, Critic

def get_gradient_penalty(D, real_x, gen_x, lambda_=10, device=torch.device('cuda')):
    """
    Calculates the gradient penalty loss term:
    λ E[ (||∇x^ D(x^)||_2 - 1)**2,
    x^ = εx_real + (1-ε)x_gen,
    λ > 0, ε ~ U(0,1)
    A way to place the K-Lipschitz constraint (||D_w|| <= K) on the critic
    """
    b = real_x.shape[0]
    eps = torch.rand(b, 1, 1, 1).to(device)
    # eps is now (b, 1, 1, 1), which makes it broadcastable with (b, c, h, w)

    interpolates = eps *real_x + (1. - eps ) *gen_x
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    d_interpolates = D(interpolates)
    grad_outputs = torch.ones_like(d_interpolates, device=device)

    gradients = torch.autograd.grad(outputs=d_interpolates,
                                    inputs=interpolates,
                                    grad_outputs=grad_outputs,
                                    create_graph=True,
                                    retain_graph=True)[0]

    gradients = gradients.view(b, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)  # gradients.norm(2, dim=1)
    gradient_penalty = lambda_ * ((gradients_norm - 1 )**2).mean()
    return gradient_penalty



def train(train_data):
    """
    Input
    - train_data: An (n_train, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]

    Returns
    - a (# of training iterations,) numpy array of WGAN critic train losses evaluated every minibatch
    - a (1000, 32, 32, 3) numpy array of samples from your model in [0, 1].
        The first 100 will be displayed, and the rest will be used to calculate the Inception score.
    """
    print(train_data.shape)

    # GAN params
    n_filters = 128
    z_dim = 128  # dimension of latent variables
    n_critic = 5  # of iterations of the critic per generator iteration

    # Train params
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda") if use_cuda else torch.device('cpu')
    n_epochs = 0
    batch_size = 256
    lr = 2e-4
    beta1, beta2 = 0, 0.9
    # c = 0.5 # the clipping parameter (not used in WGAN)
    lambda_ = 10  # Balancing coefficient for gradient penalty

    # Config for saving and loading models
    model_nr = np.random.randint(1, 1e5)
    model_path = './saved_models/'
    load_pretrained = True # Set true to load pretrained model
    save_model = False # Set true if save model periodically
    save_every = 10

    train_data = torch.from_numpy(train_data).float()
    train_data = scale(train_data, -1, 1)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    G = Generator(n_filters=n_filters, z_dim=z_dim)
    D = Critic(n_filters=n_filters)
    if use_cuda:
        G.cuda()
        D.cuda()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))

    if load_pretrained:
        print("Loading pretrained/partially trained model..")
        model_name = "gan_4460_epoch150"  # Set path to model if loading model
        checkpoint = torch.load(model_path + model_name, map_location=device)
        D.load_state_dict(checkpoint['d_state_dict'])
        G.load_state_dict(checkpoint['g_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        d_losses = checkpoint['d_losses']
        d_losses_per_gen_update = checkpoint['d_losses_per_gen_update']
        g_losses = checkpoint['g_losses']
        gps = checkpoint['gps']
        G.train(), D.train()
        print(
            f"Model loaded. [Trained for {start_epoch} epochs | Current d_loss: {round(d_losses[-1], 2)} | Current g_loss: {round(g_losses[-1], 2)}]")
        print("-" * 80)

    else:
        start_epoch = 0
        d_losses = []
        d_losses_per_gen_update = []
        g_losses = []
        gps = []

    # Training using Wasserstein distance + gradient penalty
    print("Training model..")
    train_start = time.time()
    for epoch in range(start_epoch + 1, start_epoch + n_epochs + 1):
        epoch_start = time.time()
        for i, batch in enumerate(train_loader):
            # n_critic steps of training critic | Maximize D(x) - D(G(z))
            batch = batch.cuda() if use_cuda else batch
            z = torch.randn(batch.size(0), z_dim, device=device)
            gen_samples = G(z)
            d_gen = D(gen_samples)
            d_real = D(batch)
            gradient_penalty = get_gradient_penalty(D, batch, gen_samples, lambda_=lambda_, device=device)
            d_loss = -(d_real.mean() - d_gen.mean()) + gradient_penalty

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            d_losses.append(d_loss.item())

            if i % n_critic == 0:
                # Generator update | maximixe D(G(z))
                z = torch.randn(batch.size(0), z_dim, device=device)
                gen_samples = G(z)
                d_gen = D(gen_samples)
                g_loss = -torch.mean(d_gen)
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                gps.append(gradient_penalty.item())
                g_losses.append(g_loss.item())
                d_losses_per_gen_update.append(d_loss.item() - gradient_penalty.item())
        print(f"Epoch {start_epoch}/{start_epoch + n_epochs + 1} done.")

        # Save model every now and then
        if save_model and epoch % save_every == 0:
            print(
                f'{epoch}/{start_epoch + n_epochs} epochs [{int(time.time() - epoch_start)}s] | critic loss: {round(d_loss.item(), 2)} | generator loss: {round(g_loss.item(), 2)} | gradient penalty: {round(gradient_penalty.item(), 2)}')
            model_path = "" # Set to where you want model to be saved
            save_path = f'{model_path}_{model_nr}_epoch{epoch}'
            torch.save({
                'model_nr': model_nr,
                'epoch': epoch,
                'd_state_dict': D.state_dict(),
                'g_state_dict': G.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_losses': d_losses,
                'd_losses_per_gen_update': d_losses_per_gen_update,
                'g_losses': g_losses,
                'gps': gps
            }, save_path)
    print(f'Training done in {int(time.time() - train_start)}s.')

    # Get samples from trained generator
    G.eval(), D.eval()
    with torch.no_grad():
        z = torch.randn(1000, z_dim, device=device)
        samples = G(z)
        samples = samples.permute(0, 2, 3, 1)
        samples = scale(samples, 0, 1)

    plt.plot([i for i in range(len(d_losses_per_gen_update))], np.array(d_losses_per_gen_update), label="critic loss")
    plt.plot([i for i in range(len(g_losses))], np.array(g_losses), label="generator loss")
    plt.plot([i for i in range(len(gps))], np.array(gps), label="gradient penalty")
    plt.ylabel("Loss"), plt.xlabel('Generator update')
    plt.legend()
    # plt.savefig('./results/q2_all_losses.png')
    plt.show()

    return np.array(d_losses), np.array(samples.cpu()), start_epoch + n_epochs


if __name__ == '__main__':
    # Load CIFAR-10 and show some samples
    train_data = torchvision.datasets.CIFAR10("./data", transform=torchvision.transforms.ToTensor(), train=True, download=True)
    imgs = train_data.data[:100]
    show_samples(imgs, title=f'CIFAR-10 Samples')

    # Train, then show and save generated samples
    train_data = train_data.data.transpose((0, 3, 1, 2)) / 255.0
    d_losses, samples, epochs_trained = train(train_data)
    show_samples(samples[:100] * 255.0, title=f'CIFAR-10 generated samples after {epochs_trained} epochs', fname=f'./samples/cifar10_epochs{epochs_trained}')



