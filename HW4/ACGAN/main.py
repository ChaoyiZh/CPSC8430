import torchvision
from torch.utils.data import DataLoader
from model import *
import torch
import os
from tqdm import tqdm
from torch.autograd import Variable
from pytorch_gan_metrics import get_inception_score
from torchvision import utils
import numpy as np


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

acgan_generator = ACGAN_Generator()
acgan_discriminator = ACGAN_Discriminator()
print(acgan_generator)
print(acgan_discriminator)
acgan_generator.to(device)
acgan_discriminator.to(device)

learning_rate = 0.0002
epochs = 50


def train(generator, discriminator, train_dataloader, results_dir,):
        source_criterion = nn.BCELoss()
        class_criterion = nn.NLLLoss()
        optim_generator = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        if not os.path.exists(os.path.join(results_dir,'train_generated_images_acgan')):
                os.makedirs(os.path.join(results_dir,'train_generated_images_acgan'))


        inception_score_file = open(os.path.join(results_dir, "inception_score_acgan.csv"), "w")
        inception_score_file.write('epoch, inception_score \n')

        for epoch in tqdm(range(epochs)):
                for images, labels in train_dataloader:
                        batch_size = images.shape[0]
                        real_images = Variable(images.type(torch.cuda.FloatTensor)).to(device)
                        real_labels = Variable(labels.type(torch.cuda.LongTensor)).to(device)

                        fake = torch.zeros(batch_size).to(device)
                        valid = torch.ones(batch_size).to(device)

                        optim_generator.zero_grad()
                        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (batch_size, 100))))
                        generated_labels = Variable(torch.cuda.LongTensor(np.random.randint(0, 10, batch_size)))

                        generated_images = generator(z, generated_labels)

                        validity, predicted_label = discriminator(generated_images)
                        gen_loss = 0.5 * (source_criterion(validity, valid.unsqueeze(1)) + class_criterion(
                                predicted_label, generated_labels))
                        gen_loss.backward()
                        optim_generator.step()

                        optim_discriminator.zero_grad()

                        # compute real images loss
                        real_pred, real_aux = discriminator(real_images)
                        disc_loss_real = 0.5 * (
                                        source_criterion(real_pred, valid.unsqueeze(1)) + class_criterion(real_aux,
                                                                                                          real_labels))

                        # compute fake images loss
                        fake_pred, fake_aux = discriminator(generated_images.detach())
                        disc_loss_fake = 0.5 * (
                                        source_criterion(fake_pred, fake.unsqueeze(1)) + class_criterion(fake_aux,
                                                                                                         generated_labels))

                        # compute overall discriminator loss, optimize discriminator
                        disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)
                        disc_loss.backward()
                        optim_discriminator.step()

                z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (1000, 100))))
                generated_labels = Variable(torch.cuda.LongTensor(np.random.randint(0, 10, 1000)))
                samples = generator(z, generated_labels)

                # normalize to [0, 1]
                samples = samples.mul(0.5).add(0.5)

                assert 0 <= samples.min() and samples.max() <= 1
                inception_score, inception_score_std = get_inception_score(samples)
                print("epoch: " + str(epoch) + ', inception score: ' + str(round(inception_score, 2)) + ' ± ' + str(
                        round(inception_score_std, 2)))

                samples = samples[:16].data.cpu()
                grid = utils.make_grid(samples, nrow=4)
                utils.save_image(grid, os.path.join(results_dir, 'train_generated_images_acgan/epoch_{}.png').format(str(epoch)))

                inception_score_file.write(str(epoch) + ', ' + str(round(inception_score, 2)) + '\n')

        inception_score_file.close()

def generate_images(generator, testloader, results_dir):
        for images, labels in testloader:
                z = torch.randn(16, 100).to(device)
                samples = generator(z, labels[:16].to(device))
                samples = samples.mul(0.5).add(0.5)
                samples = samples.data.cpu()
                grid = utils.make_grid(samples, nrow=4)
                print("Grid of 4*4 images saved to 'acgan_generated_images.png'.")
                utils.save_image(grid, os.path.join(results_dir, 'acgan_generated_images.png'))
                break





def load_model(model, model_filename):
    model.load_state_dict(torch.load(model_filename))

print("training DCGAN model...")
results_dir = "/home/chaoyi/workspace/course/CPSC8430/HW4/ACGAN/results"
os.makedirs(results_dir, exist_ok=True)
train(acgan_generator, acgan_discriminator, train_CIFAR10_dataloader, results_dir)


# save DCGAN to file
print("saving DCGAN model to file...")
torch.save(acgan_generator.state_dict(), os.path.join(results_dir, 'acgan_generator.pkl'))
torch.save(acgan_discriminator.state_dict(), os.path.join(results_dir, 'acgan_discriminator.pkl'))

print("loading DCGAN model...")
load_model(acgan_generator, os.path.join(results_dir,'acgan_generator.pkl'))
load_model(acgan_discriminator, os.path.join(results_dir,'acgan_discriminator.pkl'))

generate_images(acgan_generator, test_CIFAR10_dataloader, results_dir)