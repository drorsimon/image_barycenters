import os
from os.path import join

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F

import dataloader
import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.transforms as transforms
from dcgan_models import Discriminator, Encoder, Generator
from torchvision.utils import save_image
from utils import alexnet_norm, denorm




# Plot losses
def plot_loss(model1_losses, model2_losses, num_epochs, log_dir, model1='Discriminator', model2='Generator'):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    if model2_losses is not None:
        ax.set_ylim(0, max(np.max(model2_losses), np.max(model1_losses))*1.1)
    else:
        ax.set_ylim(0, np.max(model1_losses)*1.1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss values')
    plt.plot(model1_losses, label=model1)
    if model2_losses is not None:
        plt.plot(model2_losses, label=model2)
    plt.legend()

    # save figure
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    fig_path = join(log_dir, '_'.join((model1,model2)) + '_DCGAN_losses.png')
    plt.savefig(fig_path)
    plt.close()

def plot_result(generated_images, num_epoch, log_dir, fig_size=(5, 5)):
    n_rows = int(np.sqrt(generated_images.shape[0]))
    n_cols = n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for ax, img in zip(axes.flatten(), generated_images):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        # Scale to 0-255
        img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch+1)
    fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    save_fn = join(log_dir, 'shoes_DCGAN_epoch_{:d}'.format(num_epoch+1) + '.png')
    plt.savefig(save_fn)
    plt.close()

# Training methods
def train_gan(latent_dim=100, num_filters=[1024, 512, 256, 128], batch_size=128, num_epochs=100, h5_file_path='shoes_images/shoes.hdf5', save_dir='networks/', train_log_dir='dcgan_log_dir', learning_rate=0.0002, betas=(0.5, 0.999)):
    # Models
    G = Generator(latent_dim, num_filters)
    D = Discriminator(num_filters[::-1])
    G.cuda()
    D.cuda()

    # Loss function
    criterion = torch.nn.BCELoss()

    # Optimizers
    G_optimizer = optim.Adam(G.parameters(), lr=learning_rate, betas=betas, weight_decay=1e-5)
    D_optimizer = optim.Adam(D.parameters(), lr=learning_rate, betas=betas, weight_decay=1e-5)
    
    # Schedulers
    G_scheduler = optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[25,50,75])
    D_scheduler = optim.lr_scheduler.MultiStepLR(D_optimizer, milestones=[25,50,75])

    # loss arrays
    D_avg_losses = []
    G_avg_losses = []

    # Fixed noise for test
    num_test_samples = 6*6
    fixed_noise = torch.randn(num_test_samples, latent_dim, 1, 1).cuda()

    # Dataloader
    data_loader = dataloader.get_h5_dataset(path=h5_file_path, batch_size=batch_size)

    for epoch in range(num_epochs):
        D_epoch_losses = []
        G_epoch_losses = []

        for i, images in enumerate(data_loader):
            mini_batch = images.size()[0]
            x = images.cuda()

            y_real = torch.ones(mini_batch).cuda()
            y_fake = torch.zeros(mini_batch).cuda()

            # Train discriminator
            D_real_decision = D(x).squeeze()
            D_real_loss = criterion(D_real_decision, y_real)

            z = torch.randn(mini_batch, latent_dim, 1, 1)
            z = z.cuda()
            generated_images = G(z)

            D_fake_decision = D(generated_images).squeeze()
            D_fake_loss = criterion(D_fake_decision, y_fake)

            # Backprop
            D_loss = D_real_loss + D_fake_loss
            D.zero_grad()
            if i%2 == 0:  # Update discriminator only once every 2 batches
                D_loss.backward()
                D_optimizer.step()

            # Train generator
            z = torch.randn(mini_batch, latent_dim, 1, 1)
            z = z.cuda()
            generated_images = G(z)

            D_fake_decision = D(generated_images).squeeze()
            G_loss = criterion(D_fake_decision, y_real)

            # Backprop Generator
            D.zero_grad()
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # loss values
            D_epoch_losses.append(D_loss.data.item())
            G_epoch_losses.append(G_loss.data.item())

            print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                % (epoch+1, num_epochs, i+1, len(data_loader), D_loss.data.item(), G_loss.data.item()))

        D_avg_loss = torch.mean(torch.FloatTensor(D_epoch_losses)).item()
        G_avg_loss = torch.mean(torch.FloatTensor(G_epoch_losses)).item()
        D_avg_losses.append(D_avg_loss)
        G_avg_losses.append(G_avg_loss)

        # Plots
        plot_loss(D_avg_losses, G_avg_losses, num_epochs, log_dir=train_log_dir)
        
        G.eval()
        generated_images = G(fixed_noise).detach()
        generated_images = denorm(generated_images)
        G.train()
        plot_result(generated_images, epoch, log_dir=train_log_dir)

        # Save models
        torch.save(G.state_dict(), join(save_dir,'generator'))
        torch.save(D.state_dict(), join(save_dir,'discriminator'))

        # Decrease learning-rate
        G_scheduler.step()
        D_scheduler.step()

def train_encoder_with_noise(latent_dim=100, num_filters=[1024, 512, 256, 128], batch_size=128, num_epochs=100, h5_file_path='shoes_images/shoes.hdf5', save_dir='networks/', train_log_dir='dcgan_log_dir', learning_rate=0.0002, betas=(0.5, 0.999)):
    # Load generator and fix weights
    G = Generator(latent_dim, num_filters).cuda()
    generator_path = join(save_dir,'generator')
    G.load_state_dict(torch.load(generator_path))
    G.eval()
    for param in G.parameters():
        param.requires_grad = False
    
    E = Encoder(num_filters[::-1], latent_dim)
    E.cuda()

    # Loss function
    criterion = torch.nn.MSELoss()

    # Optimizer
    E_optimizer = optim.Adam(E.parameters(), lr=learning_rate, betas=betas, weight_decay=1e-5)

    E_avg_losses = []

    # Dataloader
    data_loader = dataloader.get_h5_dataset(path=h5_file_path, batch_size=batch_size)

    for epoch in range(num_epochs):
        E_losses = []

        # minibatch training
        for i, images in enumerate(data_loader):

            # generate_noise
            z = torch.randn(images.shape[0],latent_dim,1,1).cuda()
            x = G(z)

            # Train Encoder
            out_latent = E(x)
            E_loss = criterion(z, out_latent)

            # Back propagation
            E.zero_grad()
            E_loss.backward()
            E_optimizer.step()

            # loss values
            E_losses.append(E_loss.data.item())

            print('Epoch [%d/%d], Step [%d/%d], E_loss: %.4f'
                % (epoch+1, num_epochs, i+1, len(data_loader), E_loss.data.item()))

        E_avg_loss = torch.mean(torch.FloatTensor(E_losses)).item()

        # avg loss values for plot
        E_avg_losses.append(E_avg_loss)

        plot_loss(E_avg_losses, None, num_epochs, log_dir=train_log_dir, model1='Encoder', model2='')

        # Save models
        torch.save(E.state_dict(), join(save_dir,'encoder'))

def finetune_encoder_with_samples(latent_dim=100, num_filters=[1024, 512, 256, 128], batch_size=128, num_epochs=100, h5_file_path='shoes_images/shoes.hdf5', save_dir='networks/', train_log_dir='dcgan_log_dir', learning_rate=0.0002, betas=(0.5, 0.999), alpha=0.002):
    # load alexnet:
    alexnet = models.alexnet(pretrained=True).cuda()
    alexnet.eval()
    for param in alexnet.parameters():
        param.requires_grad = False

    # Load generator and fix weights
    G = Generator(latent_dim, num_filters).cuda()
    generator_path = join(save_dir,'generator')
    G.load_state_dict(torch.load(generator_path))
    G.eval()
    for param in G.parameters():
        param.requires_grad = False

    # Load encoder    
    E = Encoder(num_filters[::-1], latent_dim).cuda()
    encoder_path = join(save_dir,'encoder')
    E.load_state_dict(torch.load(encoder_path))
    E.train()

    # Loss function
    criterion = torch.nn.MSELoss()

    # Optimizers
    E_optimizer = optim.Adam(E.parameters(), lr=learning_rate, betas=betas, weight_decay=1e-5)

    E_avg_losses = []

    # Dataloader
    data_loader = dataloader.get_h5_dataset(path=h5_file_path, batch_size=batch_size)

    interpolate = lambda x: F.interpolate(x, scale_factor=4, mode='bilinear')
    get_features = lambda x: alexnet.features(alexnet_norm(interpolate(denorm(x))))
    for epoch in range(num_epochs):
        E_losses = []

        # minibatch training
        for i, images in enumerate(data_loader):

            # generate_noise
            mini_batch = images.size()[0]
            x = images.cuda()

            # Train Encoder
            out_images = G(E(x))
            E_loss = criterion(x, out_images) + alpha*criterion(get_features(x), get_features(out_images))

            # Backprop
            E.zero_grad()
            E_loss.backward()
            E_optimizer.step()

            # loss values
            E_losses.append(E_loss.data.item())

            print('Epoch [%d/%d], Step [%d/%d], E_loss: %.4f'
                % (epoch+1, num_epochs, i+1, len(data_loader), E_loss.data.item()))

        E_avg_loss = torch.mean(torch.FloatTensor(E_losses)).item()

        # avg loss values for plot
        E_avg_losses.append(E_avg_loss)

        plot_loss(E_avg_losses, None, num_epochs, log_dir=train_log_dir, model1='Encoder', model2='')

        # Save models
        torch.save(E.state_dict(), join(save_dir,'encoder'))

def test_encoder(latent_dim=100, num_filters=[1024, 512, 256, 128], batch_size=128, num_epochs=100, h5_file_path='shoes_images/shoes.hdf5', save_dir='networks/', train_log_dir='dcgan_log_dir', alpha=0.002):
    # load alexnet:
    alexnet = models.alexnet(pretrained=True).cuda()
    alexnet.eval()
    for param in alexnet.parameters():
        param.requires_grad = False

    G = Generator(latent_dim, num_filters).cuda()
    generator_path = join(save_dir,'generator')
    G.load_state_dict(torch.load(generator_path))
    G.eval()
    for param in G.parameters():
        param.requires_grad = False

    E = Encoder(num_filters[::-1], latent_dim).cuda()
    encoder_path = join(save_dir,'encoder')
    E.load_state_dict(torch.load(encoder_path))
    E.eval()
    for param in E.parameters():
        param.requires_grad = False

    # Dataloader
    data_loader = dataloader.get_h5_dataset(path=h5_file_path, batch_size=batch_size)

    interpolate = lambda x: F.interpolate(x, scale_factor=4, mode='bilinear')
    
    images = next(iter(data_loader))
    mini_batch = images.size()[0]
    x = images.cuda()
    x_features = alexnet.features(alexnet_norm(interpolate(denorm(x))))

    # Encode
    z = E(x)
    out_images = torch.stack((denorm(x),denorm(G(z))), dim=1)

    z.requires_grad_(True)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([z], lr=1e-3)
    
    for num_epoch in range(100):    
        outputs = G(z)
        # loss = criterion(outputs, x_)
        loss = criterion(x, outputs) + 0.002*criterion(x_features, alexnet.features(alexnet_norm(interpolate(denorm(outputs)))))
        z.grad = None
        loss.backward()
        optimizer.step()
    out_images = torch.cat((out_images,denorm(G(z)).unsqueeze(1)), dim=1)
            
    
    nrow = out_images.shape[1]
    out_images = out_images.reshape(-1, *x.shape[1:])
    save_image(out_images, join(train_log_dir,'encoder_images.png'), nrow=nrow, normalize=False, scale_each=False, range=(0,1))


if __name__ == '__main__':
    train_gan()
    train_encoder_with_noise()
    finetune_encoder_with_samples()
    test_encoder()
