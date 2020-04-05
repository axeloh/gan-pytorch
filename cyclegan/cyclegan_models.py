from utils import *

def conv(c_in, c_out, kernel_size, stride=2, padding=1, bn=True):
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def deconv(c_in, c_out, kernel_size, stride=2, padding=1, bn=True):
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, kernel_size, stride, padding, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class G_AB(nn.Module):
    """Generator for transfering from regular mnist to colored mnist
    Input: (b, 1, 32, 32)
    Output: (b, 1, 32, 32)
    """

    def __init__(self, n_filters=64):
        super().__init__()
        # Input is (b, 1, 32, 23)
        self.convnet = nn.Sequential(
            conv(1, n_filters, 4),  # (b, 64, 16, 16)
            nn.LeakyReLU(0.05),
            conv(n_filters, 2 * n_filters, 4),  # (b, 128, 8, 8)
            nn.LeakyReLU(0.05),
            conv(2 * n_filters, 2 * n_filters, 3, 1, 1),  # ("")
            nn.LeakyReLU(0.05),
            conv(2 * n_filters, 2 * n_filters, 3, 1, 1),  # ("")
            nn.LeakyReLU(0.05),
            deconv(2 * n_filters, n_filters, 4),  # (b, 64, 16, 16)
            nn.LeakyReLU(0.05),
            deconv(n_filters, 3, 4, bn=False),  # (b, 3, 32, 32)
            nn.Tanh()
        )

    def forward(self, x):
        return self.convnet(x)


class G_BA(nn.Module):
    """Generator for transfering from colored mnist to regular mnist
    Input: (b, 3, 32, 32)
    Output: (b, 1, 32, 32)
    """

    def __init__(self, n_filters=64):
        super().__init__()
        self.convnet = nn.Sequential(
            conv(3, n_filters, 4),  # (b, 64, 16, 16)
            nn.LeakyReLU(0.05),
            conv(n_filters, 2 * n_filters, 4),  # (b, 128, 8, 8)
            nn.LeakyReLU(0.05),
            conv(2 * n_filters, 2 * n_filters, 3, 1, 1),  # ("")
            nn.LeakyReLU(0.05),
            conv(2 * n_filters, 2 * n_filters, 3, 1, 1),  # ("")
            nn.LeakyReLU(0.05),
            deconv(2 * n_filters, n_filters, 4),  # (b, 64, 16, 16)
            nn.LeakyReLU(0.05),
            deconv(n_filters, 1, 4, bn=False),  # (b, 1, 32, 32)
            nn.Tanh()
        )

    def forward(self, x):
        return self.convnet(x)


class D_A(nn.Module):
    """
    Discriminator classifying real regular mnist from fake regular mnist
    Input: (b, 1, 32, 32)
    Output: (b, 1)
    """

    def __init__(self, n_filters=64, use_labels=False):
        super().__init__()
        self.conv1 = conv(1, n_filters, 4, bn=False)
        self.conv2 = conv(n_filters, 2 * n_filters, 4)
        self.conv3 = conv(2 * n_filters, 4 * n_filters, 4)

        n_out = 11 if use_labels else 1  # 10 classes + 1 class for fake image
        self.fc = nn.Linear(4 * 4 * (4 * n_filters), n_out)

    def forward(self, x):
        y = F.leaky_relu(self.conv1(x), 0.05)  # (b, 64, 16, 16)
        y = F.leaky_relu(self.conv2(y), 0.05)  # (b, 128, 8, 8)
        y = F.leaky_relu(self.conv3(y), 0.05)  # (b, 256, 4, 4)
        y = y.view(y.size(0), -1)  # (b, 4*4*256)
        y = F.sigmoid(self.fc(y))  # (b, n_out)
        return y


class D_B(nn.Module):
    """Discriminator classifying real colored mnist from fake colored mnist
    Input: (b, 3, 32, 32)
    Output: (b, 1)
    """

    def __init__(self, n_filters=64, use_labels=False):
        super().__init__()
        self.conv1 = conv(3, n_filters, 4, bn=False)
        self.conv2 = conv(n_filters, 2 * n_filters, 4)
        self.conv3 = conv(2 * n_filters, 4 * n_filters, 3)

        n_out = 11 if use_labels else 1
        self.fc = nn.Linear(4 * 4 * (4 * n_filters), n_out)

    def forward(self, x):
        y = F.leaky_relu(self.conv1(x), 0.05)  # (b, 64, 16, 16)
        y = F.leaky_relu(self.conv2(y), 0.05)  # (b, 128, 8, 8)
        y = F.leaky_relu(self.conv3(y), 0.05)  # (b, 256, 4, 4)
        y = y.view(y.size(0), -1)  # (b, 4*4*256)
        y = F.sigmoid(self.fc(y))  # (b, n_out)
        return y