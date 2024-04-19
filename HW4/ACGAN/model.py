from torch import nn
import torch
class ACGAN_Generator(nn.Module):
    def __init__(self):
        super(ACGAN_Generator, self).__init__()
        self.emb = nn.Embedding(10, 100)
        self.net_seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, True),
            # nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(num_features=64),
            # nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh())

    def forward(self, x, lables):
        x = torch.mul(self.emb(lables), x)
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return self.net_seq(x)

class ACGAN_Discriminator(nn.Module):
    def __init__(self):
        super(ACGAN_Discriminator, self).__init__()
        self.net_seq = nn.Sequential(
        # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
        # nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True))

        self.adv_layer = nn.Sequential(nn.Linear(512 * 4 ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(512 * 4 ** 2, 10), nn.Softmax())

    def forward(self, x):
        x = self.net_seq(x)
        x = x.view(x.shape[0], -1)
        validity = self.adv_layer(x)
        label = self.aux_layer(x)
        return validity, label