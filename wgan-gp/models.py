from utils import *


class DepthToSpace(nn.Module):
    """ DepthToSpace for changing the spatial configuration of our hidden states."""

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        # Input is (b, c, h, w)
        output = input.permute(0, 2, 3, 1)  # (b, h, w, c)
        batch_size, d_height, d_width, d_depth = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        sp1 = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in sp1]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,
                                                                                      s_depth)
        output = output.permute(0, 3, 1, 2)  # (b, c, h, w)
        return output.contiguous()


class SpaceToDepth(nn.Module):
    """ SpaceToDepth for changing the spatial configuration of our hidden states."""

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        batch_size, s_height, s_width, s_depth = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)  # (b, w, h, c)
        output = output.permute(0, 3, 1, 2)  # (b, c, w, h)
        return output.contiguous()


class Upsample_Conv2d(nn.Module):
    """ Spatial Upsampling with Nearest Neighbors """

    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.dts = DepthToSpace(block_size=2)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)

    def forward(self, x):
        x = torch.cat([x, x, x, x], dim=1)
        x = self.dts(x)
        return self.conv2d(x)


class Downsample_Conv2d(nn.Module):
    """ Spatial Downsampling with Spatial Mean Pooling """

    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.std = SpaceToDepth(block_size=2)
        self.avgpool2d = nn.AvgPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)

    def forward(self, x):
        x = self.avgpool2d(x)
        return self.conv2d(x)


class ResnetBlockUp(nn.Module):
    """ ResBlockUp used in the generator"""

    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_dim), nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
            nn.BatchNorm2d(n_filters), nn.ReLU(),
            Upsample_Conv2d(n_filters, n_filters, kernel_size, padding=1)
        )
        self.shortcut = Upsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        _x = x
        h = self.net(_x)
        shortcut = self.shortcut(x)
        return h + shortcut


class ResnetBlockDown(nn.Module):
    """ ResBlockDown used in the discriminator"""

    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
            nn.ReLU(),
            Downsample_Conv2d(n_filters, n_filters, kernel_size, padding=1)
        )
        self.shortcut = Downsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        _x = x
        h = self.net(_x)
        shortcut = self.shortcut(x)
        return h + shortcut


class Generator(nn.Module):
    def __init__(self, n_filters=128, z_dim=128):
        super().__init__()
        self.z_dim = z_dim
        self.ln = nn.Linear(z_dim, 4 * 4 * 256)
        self.net = nn.Sequential(
            ResnetBlockUp(in_dim=256, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            nn.BatchNorm2d(n_filters), nn.ReLU(),
            nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        """ Generates images from input noise z ~ N(0,1), shape: (b, z_dim) """
        img = self.ln(z)  # (b, 4*4*256)
        img = img.view(z.shape[0], 256, 4, 4)  # (b, c, h, w)
        img = self.net(img)  # (b, 3, h, w)
        return img


class Critic(nn.Module):
    """ Discriminator. Because WGAN, it works as a critic rather than a classifier"""

    def __init__(self, n_filters=128):
        super().__init__()
        self.n_filters = n_filters
        self.net = nn.Sequential(
            ResnetBlockDown(in_dim=3, n_filters=n_filters),
            ResnetBlockDown(in_dim=n_filters, n_filters=n_filters),
            ResnetBlockDown(in_dim=n_filters, n_filters=n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=(3, 3), padding=1),
        )
        self.ln = nn.Linear(n_filters * 4 * 4, 1)

    def forward(self, img):
        b = img.shape[0]  # img shape (b, 3, h, w)
        score = self.net(img)  # (b, 128, 4, 4)
        # score = torch.sum(score, dim =(2,3)) # (b,128)
        score = score.view(b, 4 * 4 * self.n_filters)  # (b, 4*4*256)
        score = self.ln(score)  # (b, 1)
        return score




