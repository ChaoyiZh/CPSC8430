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

dcgan_generator = Generator_DCGAN()
dcgan_discriminator = Discriminator_DCGAN()
dcgan_generator.to(device)
dcgan_discriminator.to(device)

learning_rate = 0.0002
epochs = 50


def train(generator, discriminator, train_dataloader,results_dir):
        loss = nn.BCELoss()
        optim_generator = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        if not os.path.exists(os.path.join(results_dir,'train_generated_images_dcgan/')):
                os.makedirs(os.path.join(results_dir, 'train_generated_images_dcgan'), exist_ok=True)

        inception_score_file = open(os.path.join(results_dir, "inception_score_dcgan.csv"), "w")
        inception_score_file.write('epoch, inception_score \n')

        for epoch in tqdm(range(epochs)):
                for real_images, _ in train_dataloader:
                        real_images = real_images.to(device)
                        z = Variable(torch.randn(batch_size, 100, 1, 1)).to(device)
                        real_labels = torch.ones(batch_size).to(device)
                        fake_labels = torch.zeros(batch_size).to(device)

                        ### train discriminator
                        # compute loss using real images
                        preds = discriminator(real_images)
                        disc_loss_real = loss(preds.flatten(), real_labels)

                        # compute loss using fake images
                        fake_images = generator(z)
                        preds = discriminator(fake_images)
                        disc_loss_fake = loss(preds.flatten(), fake_labels)

                        # optimize discriminator
                        disc_loss = disc_loss_real + disc_loss_fake
                        discriminator.zero_grad()
                        disc_loss.backward()
                        optim_discriminator.step()

                        ### train generator
                        # compute loss with fake images
                        z = Variable(torch.randn(batch_size, 100, 1, 1)).to(device)
                        fake_images = generator(z)
                        preds = discriminator(fake_images)
                        gen_loss = loss(preds.flatten(), real_labels)

                        # optimize generator
                        generator.zero_grad()
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
                utils.save_image(grid, os.path.join(results_dir, 'train_generated_images_dcgan/epoch_{}.png'.format(str(epoch))))

                inception_score_file.write(str(epoch) + ', ' + str(round(inception_score, 2)) + '\n')

        inception_score_file.close()


def generate_images(generator, results_dir):
    z = torch.randn(16, 100, 1, 1).to(device)
    samples = generator(z)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.data.cpu()
    grid = utils.make_grid(samples, nrow=4)
    print("Grid of 4*4 images saved to 'dcgan_generated_images.png'.")
    utils.save_image(grid, os.path.join(results_dir, 'dcgan_generated_images.png'))

def load_model(model, model_filename):
    model.load_state_dict(torch.load(model_filename))

print("training DCGAN model...")
results_dir = "/home/chaoyi/workspace/course/CPSC8430/HW4/DCGAN/results"
os.makedirs(results_dir, exist_ok=True)
train(dcgan_generator, dcgan_discriminator, train_CIFAR10_dataloader, results_dir)


# save DCGAN to file
print("saving DCGAN model to file...")
torch.save(dcgan_generator.state_dict(), os.path.join(results_dir, 'dcgan_generator.pkl'))
torch.save(dcgan_discriminator.state_dict(), os.path.join(results_dir, 'dcgan_discriminator.pkl'))

print("loading DCGAN model...")
load_model(dcgan_generator, os.path.join(results_dir,'dcgan_generator.pkl'))
load_model(dcgan_discriminator, os.path.join(results_dir,'dcgan_discriminator.pkl'))

generate_images(dcgan_generator, results_dir)