import sys
import argparse
import numpy as np

import torch
from torch.nn import functional as F
from torch.autograd import Variable

import DataUtils
from SS_AAE import SS_AAE


TINY = 1e-10

def onehot(values, n_classes):
    enc = np.eye(n_classes)[values].astype('float32')
    return torch.from_numpy(enc)

def sample_categorical(batch_size, n_classes):
    '''
     Sample from a categorical distribution
     of size batch_size and # of classes n_classes
     return: torch.Tensor with the sample
    '''
    values = np.random.randint(0, high=n_classes, size=batch_size)
    return onehot(values, n_classes=n_classes)

def train(epoch, model, data_loader, args):
    model.train()
    n_batches = len(data_loader)

    for batch_idx, (X, y) in enumerate(data_loader):
        X, y = Variable(X), Variable(y)
        if args.cuda:
            X, y = X.cuda(), y.cuda()
        model.zero_grad()

        batch_sz = min(args.batch_size, len(X))
        unlabeled_sz = batch_sz - args.labeled_size
        if unlabeled_sz < 0:
            unlabeled_sz = batch_sz

        ### unlabeled examples
        Xu = X[:unlabeled_sz]

        """ Reconstruction phase """
        latent_sample = torch.cat((model.Qz(Xu), model.Qy(Xu)), dim=1)
        X_sample = model.P(latent_sample)

        recon_loss = F.binary_cross_entropy(X_sample, Xu)

        recon_loss.backward()
        model.recon_solver.step()
        model.zero_grad()

        """ Regularization phase """
        # Discriminator
        model.eval()

        z_real = Variable(torch.randn(unlabeled_sz, model.z_dim))
        y_real = Variable(sample_categorical(unlabeled_sz, model.y_dim))
        if args.cuda:
            z_real = z_real.cuda()
            y_real = y_real.cuda()
        z_fake = model.Qz(Xu)
        y_fake = model.Qy(Xu)

        Dz_real = model.Dz(z_real)
        Dy_real = model.Dy(y_real)
        Dz_fake = model.Dz(z_fake)
        Dy_fake = model.Dy(y_fake)

        Dz_loss = -((Dz_real + TINY).log() + (1 - Dz_fake + TINY).log()).mean()
        Dy_loss = -((Dy_real + TINY).log() + (1 - Dy_fake + TINY).log()).mean()
        D_loss  = Dz_loss + Dy_loss

        D_loss.backward()
        model.D_solver.step()
        model.zero_grad()

        # Generator
        model.train()

        z_fake = model.Qz(Xu)
        y_fake = model.Qy(Xu)

        Dz_fake = model.Dz(z_fake)
        Dy_fake = model.Dy(y_fake)

        Gz_loss = -(Dz_fake + TINY).log().mean()
        Gy_loss = -(Dy_fake + TINY).log().mean()
        G_loss  = Gz_loss + Gy_loss

        G_loss.backward()
        model.G_solver.step()
        model.zero_grad()

        ### labeled
        if unlabeled_sz != batch_sz:
            Xl, yl = X[unlabeled_sz:], y[unlabeled_sz:].long()

            class_loss = F.cross_entropy(model.Qy(Xl), yl)

            class_loss.backward()
            model.ss_solver.step()
            model.zero_grad()

        # logging
        if batch_idx % args.log_interval == 0 or batch_idx == n_batches-1:
            endchar = '\n' if batch_idx == n_batches-1 else '\r'
            print((
                'Train Epoch: {:2} [{:6}/{:6} ({:2.0f}%)]\t ' +
                'D_loss: {:6.4f}\t G_loss: {:6.4f}\t recon_loss: {:6.4f}'
            ).format(
                epoch,
                batch_idx * args.batch_size + len(X),
                len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                D_loss.data[0], G_loss.data[0], recon_loss.data[0]
            ), end=endchar, flush=True)

def test(epoch, model, data_loader, args):
    model.eval()
    test_loss = 0

    print('Testing...', end='\r')
    for batch_idx, (X, y) in enumerate(data_loader):
        X, y = Variable(X, volatile=True), Variable(y.long(), volatile=True)
        if args.cuda:
            X, y = X.cuda(), y.cuda()

        probs = model.Qy(X)
        _, preds = probs.max(dim=1)
        test_loss += (preds != y).sum().data[0]

    test_loss /= len(data_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def main(**kwargs):
    # Initialize arguments and torch
    parser = argparse.ArgumentParser(description='SS-AAE NIDS')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='mini batch size for training (default: 1024)')
    parser.add_argument('--labeled-size', type=int, default=128, metavar='N',
                        help='labeled size for each batch (default: 256)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training (default: False)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='number of batches between logs of \
                        training status (default: 1)')
    parser.add_argument('--test-interval', type=int, default=5, metavar='N',
                        help='number of epochs between tests of \
                        model on test set (default: 5)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    n_classes = 3
    n_features, train_loader, test_loader = DataUtils.load_data(
        args.batch_size, args.cuda, shuffle=True
    )

    # Initialize model and optimizers
    model = SS_AAE(
        X_dim = n_features,
        y_dim = n_classes,
        **kwargs
    )
    if args.cuda:
        model.cuda()

    # Log model parameters
    print('Model parameters:')
    for name, value in model.parameters():
        print('{} = {}'.format(name, value))

    # Train/test
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, args)
        if epoch % args.test_interval == 0:
            test(epoch, model, test_loader, args)

if __name__ == '__main__':
    kwargs = {
        'z_dim': 5,
        'h_dim': 300,
        'recon_lr': 1e-3,
        'D_lr'    : 1e-4,
        'G_lr'    : 1e-3,
        'ss_lr'   : 1e-3
    }
    main(**kwargs)
