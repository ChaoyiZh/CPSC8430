from torch import nn
class Generator_DCGAN(nn.Module):
    def __init__(self):
        super(Generator_DCGAN, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh())

    def forward(self, x):
        return self.net(x)

class Discriminator_DCGAN(nn.Module):
    def __init__(self):
        super(Discriminator_DCGAN, self).__init__()
        self.net = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(1024),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
        nn.Sigmoid())

    def forward(self, x):
        return self.net(x)