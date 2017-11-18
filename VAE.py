import argparse
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn

import torch
import torchvision
from torch import optim
from torch.autograd import Variable
from sklearn.decomposition import PCA

import DataUtils
import VAELib

RESULTS_DIR = '../results/'
CLASSES = ['safe', 'known attack', 'unknown attack']
CLS_COLORS = ['g', 'b', 'r']

def load_data(batch_size, cuda, data_dir='../data/'):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_dataset = DataUtils.NIDDataset(
        info_csv_file=data_dir + 'train/info.csv',
        root_dir=data_dir + 'train/',
        transform=DataUtils.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False, **kwargs
    )
    test_dataset = DataUtils.NIDDataset(
        info_csv_file=data_dir + 'test/info.csv',
        root_dir=data_dir + 'test/',
        transform=DataUtils.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, **kwargs
    )
    n_features = len(train_dataset.features) - 1

    return n_features, train_loader, test_loader

def train(epoch, model, loss_function, optimizer, data_loader, args):
    model.train()
    train_loss = 0

    for batch_idx, (data, labels) in enumerate(data_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        # loss
        loss = loss_function(recon_batch, data, mu, logvar, args)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()

        # logging
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(data_loader.dataset)))

def test(epoch, model, loss_function, data_loader, args):
    model.eval()
    test_loss = 0
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, (data, labels) in enumerate(data_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(
            recon_batch, data, mu, logvar, args
        ).data[0]

        if i == 0:
            # visualize reconstruction quality
            n = min(data.size(0), 16)
            comparison = (
                data.view(args.batch_size, 1, -1, 1)[:n]
                - recon_batch.view(args.batch_size, 1, -1, 1)[:n]
            ).abs()
            torchvision.utils.save_image(
                comparison.data.cpu(),
                RESULTS_DIR + 'reconstruction_' + str(epoch) + '.png',
                nrow=n
            )

            # visualize 2d PCA of latent variables by label
            pca = PCA(n_components=2)
            mu_2d = pca.fit_transform(mu.data.numpy())
            for label, (name, color) in enumerate(zip(CLASSES, CLS_COLORS)):
                msk = labels.numpy() == label
                ax.scatter(
                    mu_2d[msk, 0], mu_2d[msk, 1],
                    c=color, marker='x', linewidths=0.1,
                    label=name
                )
                ax.legend(loc='upper left')

    fig.savefig(RESULTS_DIR + 'latent_' + str(epoch) + '.png')

    test_loss /= len(data_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def main():
    # Initialize arguments and torch
    parser = argparse.ArgumentParser(description='VAE NIDS')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training (default: False)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='number of batches between logs of \
                        training status (default: 10)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    n_features, train_loader, test_loader = load_data(
        args.batch_size, args.cuda
    )

    # Initialize model
    model = VAELib.VAE(in_dim=n_features, latent_dim=15, hidden_dim=200)
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train/test
    for epoch in range(1, args.epochs + 1):
        train(
            epoch,
            model, VAELib.loss_function, optimizer,
            train_loader, args
        )
        test(
            epoch,
            model, VAELib.loss_function,
            test_loader, args
        )

if __name__ == '__main__':
    main()
