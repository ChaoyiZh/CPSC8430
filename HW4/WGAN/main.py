import torchvision
from torch.utils.data import DataLoader
from model import *
import torch
import os
from tqdm import tqdm
from torch.autograd import Variable
from pytorch_gan_metrics import get_inception_score
from torchvision import utils


batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(32),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_CIFAR10_set = torchvision.datasets.CIFAR10(root='../cifar10/', train=True, download=True, transform=transform)
test_CIFAR10_set = torchvision.datasets.CIFAR10(root='../cifar10/', train=False, download=True, transform=transform)

train_CIFAR10_dataloader = DataLoader(train_CIFAR10_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_CIFAR10_dataloader = DataLoader(test_CIFAR10_set, batch_size=batch_size, shuffle=True, drop_last=True)

wgan_generator = WGAN_Generator()
wgan_discriminator = WGAN_Discriminator()

wgan_generator.to(device)
wgan_discriminator.to(device)

learning_rate = 0.0002
epochs = 50


def train(generator, discriminator, train_dataloader,results_dir):
        optim_generator = torch.optim.RMSprop(generator.parameters(), lr=learning_rate)
        optim_discriminator = torch.optim.RMSprop(discriminator.parameters(), lr=learning_rate)
        weight_cliping_limit = 0.01

        if not os.path.exists(os.path.join(results_dir, 'train_generated_images_WGAN/')):
                os.makedirs(os.path.join(results_dir, 'train_generated_images_WGAN'), exist_ok=True)

        inception_score_file = open(os.path.join(results_dir,"inception_score_WGAN.csv"), "w")
        inception_score_file.write('epoch, inception_score \n')
        one = torch.FloatTensor([1])
        mone = one * -1

        for epoch in tqdm(range(epochs)):
                for real_images, _ in train_dataloader:
                        real_images = real_images.to(device)

                        z = Variable(torch.randn(batch_size, 100, 1, 1)).to(device)
                        preds = discriminator(real_images)
                        fake_images = generator(z).detach()
                        disc_loss = -torch.mean(discriminator(real_images)) + torch.mean(discriminator(fake_images))

                        # optimize discriminator
                        optim_discriminator.zero_grad()
                        disc_loss.backward()
                        optim_discriminator.step()

                        for p in discriminator.parameters():
                                p.data.clamp_(-weight_cliping_limit, weight_cliping_limit)

                        z = Variable(torch.randn(batch_size, 100, 1, 1)).to(device)
                        fake_images = generator(z)
                        preds = discriminator(fake_images)
                        gen_loss = -torch.mean(preds)

                        # optimize generator
                        optim_generator.zero_grad()
                        gen_loss.backward()
                        optim_generator.step()

                # compute inception score and samples every epoch
                z = Variable(torch.randn(800, 100, 1, 1)).to(device)
                samples = generator(z)

                # normalize to [0, 1]
                samples = samples.mul(0.5).add(0.5)

                assert 0 <= samples.min() and samples.max() <= 1
                inception_score, inception_score_std = get_inception_score(samples)
                print("epoch: " + str(epoch) + ', inception score: ' + str(round(inception_score, 2)) + ' ± ' + str(
                        round(inception_score_std, 2)))

                samples = samples[:16].data.cpu()
                grid = utils.make_grid(samples, nrow=4)
                utils.save_image(grid, os.path.join(results_dir, 'train_generated_images_WGAN/epoch_{}.png'.format(str(epoch))))
                inception_score_file.write(str(epoch) + ', ' + str(round(inception_score, 2)) + '\n')

        inception_score_file.close()


def generate_images(generator, results_dir):
    z = torch.randn(16, 100, 1, 1).to(device)
    samples = generator(z)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.data.cpu()
    grid = utils.make_grid(samples, nrow=4)
    print("Grid of 4x4 images saved to 'wgan_generated_images.png'.")
    utils.save_image(grid, os.path.join(results_dir, 'wgan_generated_images.png'))

def load_model(model, model_filename):
    model.load_state_dict(torch.load(model_filename))

print("training DCGAN model...")
results_dir = "/home/chaoyi/workspace/course/CPSC8430/HW4/WGAN/results"

os.makedirs(results_dir, exist_ok=True)
train(wgan_generator, wgan_discriminator, train_CIFAR10_dataloader, results_dir)


# save DCGAN to file
print("saving WGAN model to file...")
torch.save(wgan_generator.state_dict(), os.path.join(results_dir, 'wgan_generator.pkl'))
torch.save(wgan_discriminator.state_dict(), os.path.join(results_dir, 'wgan_discriminator.pkl'))

print("loading WGAN model...")
load_model(wgan_generator, os.path.join(results_dir,'wgan_generator.pkl'))
load_model(wgan_discriminator, os.path.join(results_dir,'wgan_discriminator.pkl'))

generate_images(wgan_generator, results_dir)